#include "rdma_common.h"
#include <regex.h>
#include <time.h>  // For timestamp-based sleep functionality

static struct rdma_event_channel *cm_event_channel = NULL;
static struct rdma_cm_id *cm_client_id = NULL;
static struct ibv_pd *pd = NULL;
static struct ibv_comp_channel *io_completion_channel = NULL;
static struct ibv_cq *client_cq = NULL;
static struct ibv_qp_init_attr qp_init_attr;
static struct ibv_qp *client_qp;
static struct ibv_mr *client_metadata_mr = NULL, *client_src_mr = NULL, *client_dst_mr = NULL, *server_metadata_mr = NULL;
static struct rdma_buffer_attr client_metadata_attr, server_metadata_attr;
static struct ibv_send_wr client_send_wr, *bad_client_send_wr = NULL;
static struct ibv_recv_wr server_recv_wr, *bad_server_recv_wr = NULL;
static struct ibv_sge client_send_sge, server_recv_sge;
static char *src = NULL, *dst = NULL;
static struct TraceEvent *events = NULL;
static int event_count = 0;
static const char *trace_filename = NULL;
static struct timespec start_time;  // Reference start time for event scheduling

#define MAX_TRACE_COUNT 4000000

/*
 * Calculate time difference in microseconds between current time and start_time
 * Returns: Time difference in microseconds
 */
static uint64_t get_time_diff_us() {
    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    
    uint64_t diff_sec = current_time.tv_sec - start_time.tv_sec;
    uint64_t diff_nsec = current_time.tv_nsec - start_time.tv_nsec;
    
    // Convert total difference to microseconds
    return diff_sec * 1000000 + diff_nsec / 1000;
}

/*
 * Sleep until the target timestamp (relative to start_time) is reached
 * Parameters:
 *   target_ts_us - Target timestamp in microseconds (from trace)
 */
static void sleep_until_timestamp(uint64_t target_ts_us) {
    uint64_t current_diff = get_time_diff_us();
    
    if (current_diff < target_ts_us) {
        uint64_t sleep_us = target_ts_us - current_diff;
        struct timespec sleep_time = {
            .tv_sec = sleep_us / 1000000,
            .tv_nsec = (sleep_us % 1000000) * 1000  // Convert remaining us to nanoseconds
        };
        nanosleep(&sleep_time, NULL);
    }
}

static void usage() {
    printf("Usage: rdma_client -a <server_ip> -f <trace_file> [-p port]\n");
    exit(1);
}

static int check_src_dst() {
    return memcmp(src, dst, strlen(src));
}

static int client_prepare_connection(struct sockaddr_in *s_addr) {
    struct rdma_cm_event *cm_event = NULL;
    int ret = -1;

    cm_event_channel = rdma_create_event_channel();
    if (!cm_event_channel) {
        rdma_error("Creating cm event channel failed, errno: %d\n", -errno);
        return -errno;
    }

    ret = rdma_create_id(cm_event_channel, &cm_client_id, NULL, RDMA_PS_TCP);
    if (ret) {
        rdma_error("Creating cm id failed, errno: %d\n", -errno);
        return -errno;
    }

    ret = rdma_resolve_addr(cm_client_id, NULL, (struct sockaddr*)s_addr, 2000);
    if (ret) {
        rdma_error("Failed to resolve address, errno: %d\n", -errno);
        return -errno;
    }

    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_ADDR_RESOLVED, &cm_event);
    if (ret) {
        rdma_error("Failed to get addr resolved event, ret: %d\n", ret);
        return ret;
    }
    rdma_ack_cm_event(cm_event);

    ret = rdma_resolve_route(cm_client_id, 2000);
    if (ret) {
        rdma_error("Failed to resolve route, errno: %d\n", -errno);
        return -errno;
    }

    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_ROUTE_RESOLVED, &cm_event);
    if (ret) {
        rdma_error("Failed to get route resolved event, ret: %d\n", ret);
        return ret;
    }
    rdma_ack_cm_event(cm_event);

    printf("Trying to connect to server at : %s port: %d\n",
        inet_ntoa(s_addr->sin_addr), ntohs(s_addr->sin_port));

    pd = ibv_alloc_pd(cm_client_id->verbs);
    if (!pd) {
        rdma_error("Failed to alloc pd, errno: %d\n", -errno);
        return -errno;
    }

    io_completion_channel = ibv_create_comp_channel(cm_client_id->verbs);
    if (!io_completion_channel) {
        rdma_error("Failed to create completion channel, errno: %d\n", -errno);
        return -errno;
    }

    client_cq = ibv_create_cq(cm_client_id->verbs, CQ_CAPACITY, NULL, io_completion_channel, 0);
    if (!client_cq) {
        rdma_error("Failed to create CQ, errno: %d\n", -errno);
        return -errno;
    }

    ret = ibv_req_notify_cq(client_cq, 0);
    if (ret) {
        rdma_error("Failed to request CQ notify, errno: %d\n", -errno);
        return -errno;
    }

    bzero(&qp_init_attr, sizeof(qp_init_attr));
    qp_init_attr.cap.max_recv_sge = MAX_SGE;
    qp_init_attr.cap.max_recv_wr = MAX_WR;
    qp_init_attr.cap.max_send_sge = MAX_SGE;
    qp_init_attr.cap.max_send_wr = MAX_WR;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.recv_cq = client_cq;
    qp_init_attr.send_cq = client_cq;

    ret = rdma_create_qp(cm_client_id, pd, &qp_init_attr);
    if (ret) {
        rdma_error("Failed to create QP, errno: %d\n", -errno);
        return -errno;
    }
    client_qp = cm_client_id->qp;

    return 0;
}

static int client_pre_post_recv_buffer() {
    int ret = -1;

    server_metadata_mr = rdma_buffer_register(pd, &server_metadata_attr,
        sizeof(server_metadata_attr), IBV_ACCESS_LOCAL_WRITE);
    if (!server_metadata_mr) {
        rdma_error("Failed to setup server metadata mr\n");
        return -ENOMEM;
    }

    server_recv_sge.addr = (uint64_t)server_metadata_mr->addr;
    server_recv_sge.length = server_metadata_mr->length;
    server_recv_sge.lkey = server_metadata_mr->lkey;

    bzero(&server_recv_wr, sizeof(server_recv_wr));
    server_recv_wr.sg_list = &server_recv_sge;
    server_recv_wr.num_sge = 1;

    ret = ibv_post_recv(client_qp, &server_recv_wr, &bad_server_recv_wr);
    if (ret) {
        rdma_error("Failed to pre-post receive buffer, errno: %d\n", ret);
        return ret;
    }

    return 0;
}

static int client_connect_to_server() {
    struct rdma_conn_param conn_param;
    struct rdma_cm_event *cm_event = NULL;
    int ret = -1;

    bzero(&conn_param, sizeof(conn_param));
    conn_param.initiator_depth = 3;
    conn_param.responder_resources = 3;
    conn_param.retry_count = 3;

    ret = rdma_connect(cm_client_id, &conn_param);
    if (ret) {
        rdma_error("Failed to connect, errno: %d\n", -errno);
        return -errno;
    }

    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_ESTABLISHED, &cm_event);
    if (ret) {
        rdma_error("Failed to get established event, ret: %d\n", ret);
        return ret;
    }
    rdma_ack_cm_event(cm_event);

    printf("The client is connected successfully\n");
    return 0;
}

static int client_xchange_metadata_with_server() {
    struct ibv_wc wc;
    int ret = -1;

    src = calloc(1, SERVER_BUFFER_SIZE);
    if (!src) {
        rdma_error("Failed to allocate src/dst buffers\n");
        return -ENOMEM;
    }

    client_src_mr = rdma_buffer_register(pd, src, SERVER_BUFFER_SIZE,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!client_src_mr) {
        rdma_error("Failed to register src buffer\n");
        return -ENOMEM;
    }

    client_metadata_attr.address = (uint64_t)client_src_mr->addr;
    client_metadata_attr.length = client_src_mr->length;
    client_metadata_attr.stag.local_stag = client_src_mr->lkey;

    client_metadata_mr = rdma_buffer_register(pd, &client_metadata_attr,
        sizeof(client_metadata_attr), IBV_ACCESS_LOCAL_WRITE);
    if (!client_metadata_mr) {
        rdma_error("Failed to register client metadata\n");
        return -ENOMEM;
    }

    client_send_sge.addr = (uint64_t)client_metadata_mr->addr;
    client_send_sge.length = client_metadata_mr->length;
    client_send_sge.lkey = client_metadata_mr->lkey;

    bzero(&client_send_wr, sizeof(client_send_wr));
    client_send_wr.opcode = IBV_WR_SEND;
    client_send_wr.sg_list = &client_send_sge;
    client_send_wr.num_sge = 1;
    client_send_wr.send_flags = IBV_SEND_SIGNALED;

    ret = ibv_post_send(client_qp, &client_send_wr, &bad_client_send_wr);
    if (ret) {
        rdma_error("Failed to post send, errno: %d\n", ret);
        return ret;
    }

    ret = process_work_completion_events(io_completion_channel, &wc, 1);
    if (ret <= 0) {
        rdma_error("Failed to get send completion, ret: %d\n", ret);
        return ret;
    }

    ret = process_work_completion_events(io_completion_channel, &wc, 1);
    if (ret <= 0) {
        rdma_error("Failed to get recv completion, ret: %d\n", ret);
        return ret;
    }

    show_rdma_buffer_attr(&server_metadata_attr);
    return 0;
}

static int load_trace() {
    FILE *file = fopen(trace_filename, "r");
    if (!file) {
        rdma_error("Failed to open trace file: %s\n", trace_filename);
        return -1;
    }

    regex_t retrieve_re, offload_re;
    if (regcomp(&retrieve_re, "Retrieve start: size=([0-9]+) bytes, timestamp=([0-9]+)[[:space:]]*μs", REG_EXTENDED) != 0) {
        rdma_error("Failed to compile retrieve regex\n");
        fclose(file);
        return -1;
    }
    if (regcomp(&offload_re, "Offload start: size=([0-9]+) bytes, timestamp=([0-9]+)[[:space:]]*μs", REG_EXTENDED) != 0) {
        rdma_error("Failed to compile offload regex\n");
        regfree(&retrieve_re);
        fclose(file);
        return -1;
    }

    char line[1024];
    event_count = 0;
    events = malloc(MAX_TRACE_COUNT * sizeof(struct TraceEvent));
    if (!events) {
        rdma_error("Failed to allocate events buffer\n");
        regfree(&retrieve_re);
        regfree(&offload_re);
        fclose(file);
        return -1;
    }

    int offload_count = 0, retrieve_count = 0;

    while (fgets(line, sizeof(line), file)) {
        regmatch_t pmatch[3];
        if (regexec(&retrieve_re, line, 3, pmatch, 0) == 0) {
            if (event_count >= MAX_TRACE_COUNT) break;
            events[event_count].type = RETRIEVE;
            events[event_count].data_size = atoi(&line[pmatch[1].rm_so]);
            events[event_count].timestamp = atoll(&line[pmatch[2].rm_so]);
            event_count++;
            retrieve_count++;
        } else if (regexec(&offload_re, line, 3, pmatch, 0) == 0) {
            if (event_count >= MAX_TRACE_COUNT) break;
            events[event_count].type = OFFLOAD;
            events[event_count].data_size = atoi(&line[pmatch[1].rm_so]);
            events[event_count].timestamp = atoll(&line[pmatch[2].rm_so]);
            event_count++;
            offload_count++;
        }
    }

    uint64_t base_ts = events[0].timestamp;
    for (int i = 0; i < event_count; i++) {
        events[i].timestamp -= base_ts;
    }

    regfree(&retrieve_re);
    regfree(&offload_re);
    fclose(file);
    printf("Loaded %d events from trace file\n", event_count);
    printf("Offload events: %d, Retrieve events: %d\n", offload_count, retrieve_count);
    return 0;
}

static int process_events() {
    struct ibv_wc wc;
    int ret;

    // Record start time (reference point for trace timestamps)
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    printf("Starting event processing with timestamp synchronization\n");

    for (int i = 0; i < event_count; i++) {
        struct TraceEvent *event = &events[i];
        if (event->data_size > SERVER_BUFFER_SIZE) {
            rdma_error("Event %d size (%u) exceeds buffer capacity\n", i, event->data_size);
            continue;
        }

        // Sleep until the event's timestamp (relative to start_time)
        sleep_until_timestamp(event->timestamp);

        if (event->type == OFFLOAD) {
            memset(src, 0xAA, event->data_size);
            bzero(&client_send_wr, sizeof(client_send_wr));
            client_send_wr.opcode = IBV_WR_RDMA_WRITE;
            client_send_wr.sg_list = &client_send_sge;
            client_send_wr.num_sge = 1;
            client_send_wr.send_flags = IBV_SEND_SIGNALED;
            client_send_wr.wr.rdma.rkey = server_metadata_attr.stag.remote_stag;
            client_send_wr.wr.rdma.remote_addr = server_metadata_attr.address;
            client_send_sge.addr = (uint64_t)src;
            client_send_sge.length = event->data_size;
            client_send_sge.lkey = client_src_mr->lkey;

            ret = ibv_post_send(client_qp, &client_send_wr, &bad_client_send_wr);
        } else {
            bzero(&client_send_wr, sizeof(client_send_wr));
            client_send_wr.opcode = IBV_WR_RDMA_READ;
            client_send_wr.sg_list = &client_send_sge;
            client_send_wr.num_sge = 1;
            client_send_wr.send_flags = IBV_SEND_SIGNALED;
            client_send_wr.wr.rdma.rkey = server_metadata_attr.stag.remote_stag;
            client_send_wr.wr.rdma.remote_addr = server_metadata_attr.address;
            client_send_sge.addr = (uint64_t)src;
            client_send_sge.length = event->data_size;
            client_send_sge.lkey = client_src_mr->lkey;

            ret = ibv_post_send(client_qp, &client_send_wr, &bad_client_send_wr);
        }

        if (ret) {
            rdma_error("Failed to post %s, errno: %d\n",
                event->type == RETRIEVE ? "read" : "write", ret);
            return ret;
        }

        // Wait for completion before next event
        ret = process_work_completion_events(io_completion_channel, &wc, 1);
        if (ret <= 0) {
            rdma_error("Failed to get completion for event %d, ret: %d\n", i, ret);
            return ret;
        }
        // printf("Completed event %d: %s of %u bytes at timestamp %lu us\n",
        //     i, event->type == RETRIEVE ? "Retrieve" : "Offload",
        //     event->data_size, event->timestamp);
        if ((i + 1) % 100000 == 0) {
            printf("Processed %d events so far\n", i + 1);
        }
    }

    return 0;
}

static int client_cleanup() {
    if (events) free(events);
    if (src) free(src);
    if (dst) free(dst);
    if (client_src_mr) rdma_buffer_deregister(client_src_mr);
    if (client_metadata_mr) rdma_buffer_deregister(client_metadata_mr);
    if (server_metadata_mr) rdma_buffer_deregister(server_metadata_mr);
    if (client_qp) rdma_destroy_qp(cm_client_id);
    if (client_cq) ibv_destroy_cq(client_cq);
    if (io_completion_channel) ibv_destroy_comp_channel(io_completion_channel);
    if (pd) ibv_dealloc_pd(pd);
    if (cm_client_id) rdma_destroy_id(cm_client_id);
    if (cm_event_channel) rdma_destroy_event_channel(cm_event_channel);

    printf("Client resource cleanup is complete\n");
    return 0;
}

int main(int argc, char *argv[]) {
    int ret, option;
    struct sockaddr_in s_addr;
    bzero(&s_addr, sizeof(s_addr));
    s_addr.sin_family = AF_INET;
    s_addr.sin_port = htons(DEFAULT_RDMA_PORT);

    while ((option = getopt(argc, argv, "a:p:f:")) != -1) {
        switch (option) {
            case 'a':
                ret = get_addr(optarg, (struct sockaddr *)&s_addr);
                if (ret) {
                    rdma_error("Invalid IP address: %s\n", optarg);
                    return ret;
                }
                break;
            case 'p':
                s_addr.sin_port = htons(strtol(optarg, NULL, 0));
                break;
            case 'f':
                trace_filename = optarg;
                break;
            default:
                usage();
                break;
        }
    }

    if (!s_addr.sin_port) {
	    s_addr.sin_port = htons(DEFAULT_RDMA_PORT);
	}

    if (!trace_filename) {
        rdma_error("Trace file not specified (-f)\n");
        usage();
    }

    if (load_trace() != 0) {
        return 1;
    }

    if (client_prepare_connection(&s_addr) != 0) {
        client_cleanup();
        return 1;
    }

    if (client_pre_post_recv_buffer() != 0) {
        client_cleanup();
        return 1;
    }

    if (client_connect_to_server() != 0) {
        client_cleanup();
        return 1;
    }

    if (client_xchange_metadata_with_server() != 0) {
        client_cleanup();
        return 1;
    }

    if (process_events() != 0) {
        client_cleanup();
        return 1;
    }

    client_cleanup();
    return 0;
}