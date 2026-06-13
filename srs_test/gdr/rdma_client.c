#include "rdma_common.h"
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <arpa/inet.h>
#include <signal.h>

static const size_t DATA_SIZE = 128ULL * 1024ULL * 1024ULL;

static struct rdma_event_channel *cm_event_channel = NULL;
static struct rdma_cm_id *cm_client_id = NULL;
static struct ibv_pd *pd = NULL;
static struct ibv_comp_channel *io_completion_channel = NULL;
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
static volatile int stop_loop = 0;

static void sigint_handler(int signum) {
    stop_loop = 1;
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

    io_completion_channel = ibv_create_comp_channel(cm_client_id->verbs);
    if (!io_completion_channel) return -ENOMEM;

    client_cq = ibv_create_cq(cm_client_id->verbs, CQ_CAPACITY, NULL, io_completion_channel, 0);
    if (!client_cq) return -ENOMEM;
    ret = ibv_req_notify_cq(client_cq, 0);
    if (ret) return -errno;

    bzero(&qp_init_attr, sizeof qp_init_attr);
    qp_init_attr.cap.max_recv_sge = MAX_SGE;
    qp_init_attr.cap.max_recv_wr = MAX_WR;
    qp_init_attr.cap.max_send_sge = MAX_SGE;
    qp_init_attr.cap.max_send_wr = MAX_WR;
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
    server_metadata_mr = rdma_buffer_register(pd, &server_metadata_attr, sizeof(server_metadata_attr), IBV_ACCESS_LOCAL_WRITE);
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
    struct ibv_wc wc[2];
    int ret = -1;

    client_src_mr = rdma_buffer_register(pd, src, DATA_SIZE,
                                        IBV_ACCESS_LOCAL_WRITE |
                                        IBV_ACCESS_REMOTE_READ |
                                        IBV_ACCESS_REMOTE_WRITE);
    if (!client_src_mr) return -ENOMEM;

    client_metadata_attr.address = (uint64_t) client_src_mr->addr;
    client_metadata_attr.length = client_src_mr->length;
    client_metadata_attr.stag.local_stag = client_src_mr->lkey;

    client_metadata_mr = rdma_buffer_register(pd, &client_metadata_attr,
                                              sizeof(client_metadata_attr),
                                              IBV_ACCESS_LOCAL_WRITE);
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

    ret = process_work_completion_events(io_completion_channel, wc, 2);
    if (ret != 2) return ret;

    show_rdma_buffer_attr(&server_metadata_attr);

    return 0;
}

static int client_remote_memory_ops()
{
    struct ibv_wc wc;
    int ret = -1;
    size_t write_size = DATA_SIZE;
    size_t total_bytes = 0;

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    client_send_sge.addr = (uint64_t) client_src_mr->addr;
    client_send_sge.length = write_size;
    client_send_sge.lkey = client_src_mr->lkey;

    while (!stop_loop) {
        bzero(&client_send_wr, sizeof(client_send_wr));
        client_send_wr.sg_list = &client_send_sge;
        client_send_wr.num_sge = 1;
        client_send_wr.opcode = IBV_WR_RDMA_WRITE;
        client_send_wr.send_flags = IBV_SEND_SIGNALED;
        client_send_wr.wr.rdma.rkey = server_metadata_attr.stag.remote_stag;
        client_send_wr.wr.rdma.remote_addr = server_metadata_attr.address;

        ret = ibv_post_send(client_qp, &client_send_wr, &bad_client_send_wr);
        if (ret) return -errno;

        ret = process_work_completion_events(io_completion_channel, &wc, 1);
        if (ret != 1) return ret;

        total_bytes += write_size;
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    double elapsed_sec = (end_time.tv_sec - start_time.tv_sec) +
                         (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    double throughput_gb_s = (double)total_bytes / (1024.0 * 1024.0 * 1024.0) / elapsed_sec;
    printf("Client side %zuB WRITE loop ended\n", write_size);
    printf("Total data written: %.2f MB\n", (double)total_bytes / (1024.0 * 1024.0));
    printf("Elapsed time: %.2f s\n", elapsed_sec);
    printf("Average throughput: %.2f GB/s\n", throughput_gb_s);

    return 0;
}

static int client_disconnect_and_clean()
{
    struct rdma_cm_event *cm_event = NULL;
    int ret = -1;

    rdma_disconnect(cm_client_id);

    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_DISCONNECTED, &cm_event);
    if (!ret) rdma_ack_cm_event(cm_event);

    rdma_destroy_qp(cm_client_id);
    rdma_destroy_id(cm_client_id);

    ibv_destroy_cq(client_cq);
    ibv_destroy_comp_channel(io_completion_channel);

    rdma_buffer_deregister(server_metadata_mr);
    rdma_buffer_deregister(client_metadata_mr);
    rdma_buffer_deregister(client_src_mr);

    free(src);

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

    src = calloc(DATA_SIZE, 1);
    if (!src) return -ENOMEM;

    while ((option = getopt(argc, argv, "a:p:")) != -1) {
        switch (option) {
            case 'a':
                ret = get_addr(optarg, (struct sockaddr*)&server_sockaddr);
                if (ret) return ret;
                break;
            case 'p':
                server_sockaddr.sin_port = htons(strtol(optarg, NULL, 0));
                break;
            default:
                fprintf(stderr, "Usage: %s -a <server_addr> [-p <port>]\n", argv[0]);
                return 1;
        }
    }

    if (!server_sockaddr.sin_port)
        server_sockaddr.sin_port = htons(DEFAULT_RDMA_PORT);

    printf("Trying to connect to server at : %s port: %d\n",
           inet_ntoa(server_sockaddr.sin_addr),
           ntohs(server_sockaddr.sin_port));

    signal(SIGINT, sigint_handler);

    ret = client_prepare_connection(&server_sockaddr);
    if (ret) return ret;

    ret = client_pre_post_recv_buffer();
    if (ret) return ret;

    ret = client_connect_to_server();
    if (ret) return ret;

    ret = client_xchange_metadata_with_server();
    if (ret) return ret;

    ret = client_remote_memory_ops();

    ret = client_disconnect_and_clean();
    return ret;
}