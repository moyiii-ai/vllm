#include "rdma_common.h"
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <arpa/inet.h>
#include <signal.h>
#include <poll.h>
#include <cuda_runtime.h>

static size_t data_size = (32ULL * 1024ULL * 1024ULL);
static int operation_type; // 1 = WRITE, 0 = READ
static int gpu_device_id = -1;  // -1 = CPU memory, >= 0 = GPU memory
static int use_gpu = 0;  // Flag to use GPU memory
static long num_iterations = -1;  // Number of iterations, -1 = infinite
static pid_t notify_ready_pid = -1;  // if >0, signal this PID when handshake complete
static int ready_sleep_sec = 1;      // sleep before notifying readiness
static int tx_depth = 128;           // outstanding WRs, similar to ib_write_bw --tx-depth
static int cq_mod = 64;              // CQ moderation, signal every cq_mod WRs
static long report_every = 400;      // report throughput every N iterations (0 = disable)
static int post_list = 64;           // perftest --post_list: number of WRs per ibv_post_send() doorbell
static long warmup_iters = 0;        // perftest-like warmup iterations (not counted in final stats)

static struct rdma_event_channel *cm_event_channel = NULL;
static struct rdma_cm_id *cm_client_id = NULL;
static struct ibv_pd *pd = NULL;
static struct ibv_cq *client_cq = NULL;
static struct ibv_qp_init_attr qp_init_attr;
static struct ibv_qp *client_qp;

static struct ibv_mr *client_metadata_mr = NULL, 
                     *client_src_mr = NULL, 
                     *server_metadata_mr = NULL;
static struct rdma_buffer_attr client_metadata_attr, server_metadata_attr;
static struct ibv_send_wr client_send_wr, *bad_client_send_wr = NULL;
static struct ibv_recv_wr server_recv_wr, *bad_server_recv_wr = NULL;
static struct ibv_sge client_send_sge, server_recv_sge;

static char *src = NULL;
static volatile sig_atomic_t start_transfer = 0;  // Flag to start transfer after handshake

FILE *log_file = NULL;  // Global log file pointer

static void sigusr1_handler(int signum) {
    (void)signum;
    start_transfer = 1;
}

static void notify_script_ready(void) {
    if (notify_ready_pid <= 0) return;
    if (ready_sleep_sec > 0) sleep((unsigned int)ready_sleep_sec);
    if (kill(notify_ready_pid, SIGUSR1) != 0) {
        fprintf(stderr, "Warning: failed to signal ready to PID %d: %s\n",
                (int)notify_ready_pid, strerror(errno));
    } else {
        printf("Signaled ready to PID %d (SIGUSR1)\n", (int)notify_ready_pid);
        fflush(stdout);
    }
}

/* Allocate memory (CPU or GPU) for RDMA operations */
static int allocate_client_memory()
{
    if (use_gpu) {
        cudaError_t err = cudaSetDevice(gpu_device_id);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", gpu_device_id, cudaGetErrorString(err));
            return -1;
        }
        
        err = cudaMalloc((void**)&src, data_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        err = cudaMemset(src, 0, data_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
            cudaFree(src);
            return -1;
        }
        
        printf("GPU memory allocated on GPU%d: %p (size: %zu MB)\n", 
               gpu_device_id, src, data_size / (1024 * 1024));
    } else {
        src = calloc(data_size, 1);
        if (!src) return -ENOMEM;
        printf("CPU memory allocated: %p (size: %zu MB)\n", 
               src, data_size / (1024 * 1024));
    }
    return 0;
}

/* Free allocated memory */
static void free_client_memory()
{
    if (use_gpu && src) {
        cudaSetDevice(gpu_device_id);
        cudaFree(src);
        src = NULL;
    } else if (src) {
        free(src);
        src = NULL;
    }
}

static int client_prepare_connection(struct sockaddr_in *s_addr)
{
    struct rdma_cm_event *cm_event = NULL;
    int ret = -1;

    cm_event_channel = rdma_create_event_channel();
    if (!cm_event_channel) return -ENOMEM;

    ret = rdma_create_id(cm_event_channel, &cm_client_id, NULL, RDMA_PS_TCP);
    if (ret) return -errno;

    ret = rdma_resolve_addr(cm_client_id, NULL, (struct sockaddr*) s_addr, 2000);
    if (ret) return -errno;
    ret  = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_ADDR_RESOLVED, &cm_event);
    if (ret) return ret;
    ret = rdma_ack_cm_event(cm_event);
    if (ret) return -errno;
    ret = rdma_resolve_route(cm_client_id, 2000);
    if (ret) return -errno;
    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_ROUTE_RESOLVED, &cm_event);
    if (ret) return ret;
    ret = rdma_ack_cm_event(cm_event);
    if (ret) return -errno;
    pd = ibv_alloc_pd(cm_client_id->verbs);
    if (!pd) return -ENOMEM;

    int cq_capacity = CQ_CAPACITY;
    if (tx_depth > 0) {
        int desired = tx_depth + 64;
        if (desired > cq_capacity) cq_capacity = desired;
    }
    // perftest/ib_write_bw style: poll CQ directly, no completion channel
    client_cq = ibv_create_cq(cm_client_id->verbs, cq_capacity, NULL, NULL, 0);
    if (!client_cq) return -ENOMEM;

    bzero(&qp_init_attr, sizeof qp_init_attr);
    qp_init_attr.cap.max_recv_sge = MAX_SGE;
    qp_init_attr.cap.max_recv_wr = MAX_WR;
    qp_init_attr.cap.max_send_sge = MAX_SGE;
    qp_init_attr.cap.max_send_wr = MAX_WR;
    if (tx_depth > 0) {
        int desired = tx_depth + 64;
        if (desired > qp_init_attr.cap.max_send_wr) qp_init_attr.cap.max_send_wr = desired;
    }
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.recv_cq = client_cq;
    qp_init_attr.send_cq = client_cq;

    ret = rdma_create_qp(cm_client_id, pd, &qp_init_attr);
    if (ret) return -errno;
    client_qp = cm_client_id->qp;
    return 0;
}

static int client_pre_post_recv_buffer()
{
    int ret = -1;
    server_metadata_mr = rdma_buffer_register(pd, &server_metadata_attr, sizeof(server_metadata_attr),
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!server_metadata_mr) return -ENOMEM;

    server_recv_sge.addr = (uint64_t) server_metadata_mr->addr;
    server_recv_sge.length = (uint32_t) server_metadata_mr->length;
    server_recv_sge.lkey = server_metadata_mr->lkey;

    bzero(&server_recv_wr, sizeof(server_recv_wr));
    server_recv_wr.sg_list = &server_recv_sge;
    server_recv_wr.num_sge = 1;

    ret = ibv_post_recv(client_qp, &server_recv_wr, &bad_server_recv_wr);
    if (ret) return ret;

    return 0;
}

static int client_connect_to_server()
{
    struct rdma_conn_param conn_param;
    struct rdma_cm_event *cm_event = NULL;
    int ret = -1;

    bzero(&conn_param, sizeof(conn_param));
    conn_param.initiator_depth = 3;
    conn_param.responder_resources = 3;
    conn_param.retry_count = 3;

    ret = rdma_connect(cm_client_id, &conn_param);
    if (ret) return -errno;

    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_ESTABLISHED, &cm_event);
    if (ret) return ret;
    ret = rdma_ack_cm_event(cm_event);
    if (ret) return -errno;

    printf("The client is connected successfully\n");
    return 0;
}

static int client_xchange_metadata_with_server()
{
    int ret = -1;

    client_src_mr = rdma_buffer_register(pd, src, (uint32_t)data_size,
                                        IBV_ACCESS_LOCAL_WRITE |
                                        IBV_ACCESS_REMOTE_READ |
                                        IBV_ACCESS_REMOTE_WRITE);
    if (!client_src_mr) return -ENOMEM;

    client_metadata_attr.address = (uint64_t) client_src_mr->addr;
    client_metadata_attr.length = client_src_mr->length;
    client_metadata_attr.stag.local_stag = client_src_mr->lkey;

    client_metadata_mr = rdma_buffer_register(pd, &client_metadata_attr,
                                              sizeof(client_metadata_attr),
                                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!client_metadata_mr) return -ENOMEM;

    client_send_sge.addr = (uint64_t) client_metadata_mr->addr;
    client_send_sge.length = client_metadata_mr->length;
    client_send_sge.lkey = client_metadata_mr->lkey;

    bzero(&client_send_wr, sizeof(client_send_wr));
    client_send_wr.sg_list = &client_send_sge;
    client_send_wr.num_sge = 1;
    client_send_wr.opcode = IBV_WR_SEND;
    client_send_wr.send_flags = IBV_SEND_SIGNALED;

    ret = ibv_post_send(client_qp, &client_send_wr, &bad_client_send_wr);
    if (ret) return -errno;

    // Poll CQ until we get both SEND and RECV completions
    int got = 0;
    struct ibv_wc wc[8];
    while (got < 2) {
        int n = ibv_poll_cq(client_cq, (int)(sizeof(wc) / sizeof(wc[0])), wc);
        if (n < 0) return n;
        if (n == 0) continue;
        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                rdma_error("Work completion (WC) error: %s\n", ibv_wc_status_str(wc[i].status));
                return -(wc[i].status);
            }
            if (wc[i].opcode == IBV_WC_SEND || wc[i].opcode == IBV_WC_RECV) {
                got++;
            }
        }
    }

    show_rdma_buffer_attr(&server_metadata_attr);

    return 0;
}

static int client_remote_memory_ops()
{
    int ret = 0;
    const size_t transfer_size = data_size;
    const long iters = num_iterations;
    long completed_ops = 0;
    long posted_ops = 0;
    int outstanding = 0;

    if (cq_mod <= 0) cq_mod = 1;
    if (tx_depth <= 0) tx_depth = 1;
    if (post_list <= 0) post_list = 1;
    if (post_list > tx_depth) post_list = tx_depth;
    if (warmup_iters < 0) warmup_iters = 0;

    struct timespec start_time, end_time, signal_recv_time;
    clock_gettime(CLOCK_MONOTONIC, &signal_recv_time);

    client_send_sge.addr = (uint64_t) client_src_mr->addr;
    client_send_sge.length = (uint32_t)transfer_size;
    client_send_sge.lkey = client_src_mr->lkey;

    size_t total_bytes = 0;
    size_t measured_bytes = 0;
    long measured_ops = 0;
    int measurement_started = (warmup_iters == 0);
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // CQ moderation accounting: each signaled WR carries the number of ops it represents
    uint64_t ops_since_signal = 0;
    struct ibv_wc wc[32];

    // Pre-allocate WR templates (perftest-style) and post in a chained batch
    struct ibv_send_wr *wr_pool = calloc((size_t)tx_depth, sizeof(*wr_pool));
    if (!wr_pool) return -ENOMEM;
    for (int i = 0; i < tx_depth; i++) {
        wr_pool[i].sg_list = &client_send_sge;
        wr_pool[i].num_sge = 1;
        wr_pool[i].opcode = operation_type ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
        wr_pool[i].wr.rdma.rkey = server_metadata_attr.stag.remote_stag;
        wr_pool[i].wr.rdma.remote_addr = server_metadata_attr.address;
        wr_pool[i].send_flags = 0;
        wr_pool[i].wr_id = 0;
        wr_pool[i].next = NULL;
    }
    int wr_idx = 0;

    struct timespec last_report_time;
    clock_gettime(CLOCK_MONOTONIC, &last_report_time);
    size_t last_report_bytes = 0;
    long last_report_ops = 0;
    long last_report_bucket = 0;

    while (iters < 0 || completed_ops < iters) {
        // perftest credit model: keep pipeline full (tx_depth), post in chunks (post_list)
        while (outstanding < tx_depth && (iters < 0 || posted_ops < iters)) {
            int remaining_depth = tx_depth - outstanding;
            int remaining_iters = (iters < 0) ? post_list : (int)((iters - posted_ops) > post_list ? post_list : (iters - posted_ops));
            int batch = remaining_depth < post_list ? remaining_depth : post_list;
            if (batch > remaining_iters) batch = remaining_iters;
            if (batch <= 0) break;

            struct ibv_send_wr *first = NULL, *prev = NULL;
            for (int j = 0; j < batch; j++) {
                struct ibv_send_wr *wr = &wr_pool[wr_idx];
                wr_idx++;
                if (wr_idx == tx_depth) wr_idx = 0;

                wr->next = NULL;
                wr->send_flags = 0;
                wr->wr_id = 0;

                ops_since_signal++;
                int must_signal = 0;
                if ((ops_since_signal % (uint64_t)cq_mod) == 0) must_signal = 1;
                if (iters >= 0 && (posted_ops + j) == (iters - 1)) must_signal = 1; // ensure tail is signaled
                if (must_signal) {
                    wr->send_flags = IBV_SEND_SIGNALED;
                    wr->wr_id = ops_since_signal; // completion covers ops_since_signal WRs
                    ops_since_signal = 0;
                }

                if (!first) first = wr;
                else prev->next = wr;
                prev = wr;
            }

            ret = ibv_post_send(client_qp, first, &bad_client_send_wr);
            if (ret) {
                free(wr_pool);
                return -errno;
            }
            posted_ops += batch;
            outstanding += batch;
        }

        // Busy poll CQ (perftest style)
        int n = ibv_poll_cq(client_cq, (int)(sizeof(wc) / sizeof(wc[0])), wc);
        if (n < 0) {
            rdma_error("ibv_poll_cq failed: %d\n", n);
            free(wr_pool);
            return n;
        }
        if (n == 0) continue;

        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                rdma_error("Work completion (WC) error: %s, wr_id=%lu\n",
                           ibv_wc_status_str(wc[i].status), (unsigned long)wc[i].wr_id);
                free(wr_pool);
                return -(wc[i].status);
            }
            uint64_t ops_done = wc[i].wr_id ? wc[i].wr_id : 1;
            completed_ops += (long)ops_done;
            outstanding -= (int)ops_done;
            total_bytes += (size_t)ops_done * transfer_size;

            // Warmup handling: start measuring after warmup_iters are completed
            if (!measurement_started && completed_ops >= warmup_iters) {
                measurement_started = 1;
                measured_ops = 0;
                measured_bytes = 0;
                clock_gettime(CLOCK_MONOTONIC, &start_time);
                last_report_time = start_time;
                last_report_ops = 0;
                last_report_bytes = 0;
                last_report_bucket = 0;
            }
            if (measurement_started) {
                measured_ops += (long)ops_done;
                measured_bytes += (size_t)ops_done * transfer_size;
            }

            if (report_every > 0) {
                if (!measurement_started) continue;
                long bucket = measured_ops / report_every;
                if (bucket > last_report_bucket) {
                    struct timespec now;
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    double interval = (now.tv_sec - last_report_time.tv_sec) +
                                      (now.tv_nsec - last_report_time.tv_nsec) / 1e9;
                    size_t bytes_delta = measured_bytes - last_report_bytes;
                    long ops_delta = measured_ops - last_report_ops;
                    double gbps = interval > 0 ? (double)bytes_delta / interval / (1024.0 * 1024.0 * 1024.0) : 0.0;
                    printf("[progress] iters=%ld (+%ld), interval=%.3fs, throughput=%.2f GB/s\n",
                           measured_ops, ops_delta, interval, gbps);
                    fflush(stdout);
                    if (log_file) {
                        fprintf(log_file, "[progress] iters=%ld (+%ld), interval=%.3fs, throughput=%.2f GB/s\n",
                                measured_ops, ops_delta, interval, gbps);
                        fflush(log_file);
                    }
                    last_report_bucket = bucket;
                    last_report_time = now;
                    last_report_bytes = measured_bytes;
                    last_report_ops = measured_ops;
                }
            }
        }
    }

    free(wr_pool);

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    if (!measurement_started) {
        // If warmup_iters >= total iters, treat all as measured
        measurement_started = 1;
        measured_ops = completed_ops;
        measured_bytes = total_bytes;
        start_time = signal_recv_time;
    }
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    double gb = (double)measured_bytes / (1024.0 * 1024.0 * 1024.0);

    printf("\n=== RDMA Client Results ===\n");
    printf("Total iterations: %ld\n", measured_ops);
    printf("Message size: %zu bytes\n", transfer_size);
    printf("tx_depth: %d, cq_mod: %d, post_list: %d, warmup: %ld\n",
           tx_depth, cq_mod, post_list, warmup_iters);
    printf("Total data transferred: %.2f GB\n", gb);
    printf("Measured time: %.3f seconds\n", elapsed_time);
    printf("Throughput: %.2f GB/s\n", gb / elapsed_time);
    if (log_file) {
        fprintf(log_file, "\n=== RDMA Client Results ===\n");
        fprintf(log_file, "Total iterations: %ld\n", measured_ops);
        fprintf(log_file, "Message size: %zu bytes\n", transfer_size);
        fprintf(log_file, "tx_depth: %d, cq_mod: %d, post_list: %d, warmup: %ld\n",
                tx_depth, cq_mod, post_list, warmup_iters);
        fprintf(log_file, "Total data transferred: %.2f GB\n", gb);
        fprintf(log_file, "Measured time: %.3f seconds\n", elapsed_time);
        fprintf(log_file, "Throughput: %.2f GB/s\n", gb / elapsed_time);
        fflush(log_file);
    }

    (void)start_time;
    return 0;
}

static int client_disconnect_and_clean()
{
    struct rdma_cm_event *cm_event = NULL;
    int ret = -1;

    rdma_disconnect(cm_client_id);

    // Avoid hanging forever waiting for DISCONNECTED on error paths
    struct pollfd pfd;
    pfd.fd = cm_event_channel->fd;
    pfd.events = POLLIN;
    int pr = poll(&pfd, 1, 2000 /* ms */);
    if (pr > 0 && (pfd.revents & POLLIN)) {
        ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_DISCONNECTED, &cm_event);
        if (!ret) rdma_ack_cm_event(cm_event);
    } else {
        fprintf(stderr, "Warning: timed out waiting for RDMA_CM_EVENT_DISCONNECTED; continuing cleanup\n");
    }

    rdma_destroy_qp(cm_client_id);
    rdma_destroy_id(cm_client_id);

    ibv_destroy_cq(client_cq);

    rdma_buffer_deregister(server_metadata_mr);
    rdma_buffer_deregister(client_metadata_mr);
    rdma_buffer_deregister(client_src_mr);

    free_client_memory();

    ibv_dealloc_pd(pd);
    rdma_destroy_event_channel(cm_event_channel);
    return 0;
}

int main(int argc, char **argv)
{
    struct sockaddr_in server_sockaddr;
    int ret, option;
    bzero(&server_sockaddr, sizeof server_sockaddr);
    server_sockaddr.sin_family = AF_INET;
    server_sockaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    while ((option = getopt(argc, argv, "a:p:rwl:g:i:P:t:S:q:C:R:B:W:")) != -1) {
        switch (option) {
            case 'a':
                ret = get_addr(optarg, (struct sockaddr*)&server_sockaddr);
                if (ret) return ret;
                break;
            case 'p':
                server_sockaddr.sin_port = htons(strtol(optarg, NULL, 0));
                break;
            case 'r':
                operation_type = 0;
                break;
            case 'w':
                operation_type = 1;
                break;
            case 'l':
                log_file = fopen(optarg, "w");
                if (log_file == NULL) {
                    fprintf(stderr, "Failed to open log file %s: %s\n", 
                            optarg, strerror(errno));
                    return 1;
                }
                break;
            case 'g':
                gpu_device_id = atoi(optarg);
                use_gpu = 1;
                break;
            case 'i':
                num_iterations = strtol(optarg, NULL, 0);
                break;
            case 'P':
                notify_ready_pid = (pid_t)strtol(optarg, NULL, 0);
                break;
            case 't':
                ready_sleep_sec = (int)strtol(optarg, NULL, 0);
                if (ready_sleep_sec < 0) ready_sleep_sec = 0;
                break;
            case 'S':
                data_size = (size_t)strtoull(optarg, NULL, 0);
                if (data_size == 0) data_size = 1;
                break;
            case 'q':
                tx_depth = (int)strtol(optarg, NULL, 0);
                break;
            case 'C':
                cq_mod = (int)strtol(optarg, NULL, 0);
                break;
            case 'R':
                report_every = strtol(optarg, NULL, 0);
                if (report_every < 0) report_every = 0;
                break;
            case 'B':
                post_list = (int)strtol(optarg, NULL, 0);
                break;
            case 'W':
                warmup_iters = strtol(optarg, NULL, 0);
                if (warmup_iters < 0) warmup_iters = 0;
                break;
            default:
                fprintf(stderr, "Usage: %s -a <server_addr> [-p <port>] [-r|-w] [-g <gpu_id>] [-i <iterations>] [-S <msg_size_bytes>] [-q <tx_depth>] [-C <cq_mod>] [-B <post_list>] [-W <warmup_iters>] [-R <report_every_iters>] [-l <log_file>] [-P <notify_pid>] [-t <ready_sleep_sec>]\n", argv[0]);
                return 1;
        }
    }

    /* Allocate memory before any RDMA operations */
    if (allocate_client_memory() != 0) {
        fprintf(stderr, "Failed to allocate client memory\n");
        return -1;
    }

    if (!server_sockaddr.sin_port)
        server_sockaddr.sin_port = htons(DEFAULT_RDMA_PORT);

    printf("Trying to connect to server at : %s port: %d\n",
           inet_ntoa(server_sockaddr.sin_addr),
           ntohs(server_sockaddr.sin_port));

    signal(SIGUSR1, sigusr1_handler);

    ret = client_prepare_connection(&server_sockaddr);
    if (ret) return ret;

    ret = client_pre_post_recv_buffer();
    if (ret) return ret;

    ret = client_connect_to_server();
    if (ret) return ret;

    ret = client_xchange_metadata_with_server();
    if (ret) return ret;

    // Handshake complete: optionally notify orchestrating script
    printf("Handshake complete. Ready to transfer.\n");
    fflush(stdout);
    notify_script_ready();

    // Wait for SIGUSR1 signal before starting transfer
    while (!start_transfer) pause();
    printf("Received start signal. Starting data transfer...\n");

    int ret_ops = client_remote_memory_ops();
    int ret_clean = client_disconnect_and_clean();
    return ret_ops ? ret_ops : ret_clean;
}
