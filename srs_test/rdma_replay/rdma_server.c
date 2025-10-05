#include "rdma_common.h"

static struct rdma_event_channel *cm_event_channel = NULL;
static struct rdma_cm_id *listener_id = NULL, *cm_server_id = NULL;
static struct ibv_pd *pd = NULL;
static struct ibv_comp_channel *io_completion_channel = NULL;
static struct ibv_cq *server_cq = NULL;
static struct ibv_qp_init_attr qp_init_attr;
static struct ibv_qp *server_qp = NULL;
static struct ibv_mr *server_metadata_mr = NULL, *server_buffer_mr = NULL;
static struct ibv_mr *client_metadata_mr = NULL;
static struct rdma_buffer_attr server_metadata_attr, client_metadata_attr;
static struct ibv_send_wr server_send_wr, *bad_server_send_wr = NULL;
static struct ibv_recv_wr client_recv_wr, *bad_client_recv_wr = NULL;
static struct ibv_sge server_send_sge, client_recv_sge;

static void usage() {
    printf("Usage: rdma_server [-p port]\n");
    exit(1);
}

static int start_rdma_server(struct sockaddr_in *server_addr) {
    struct rdma_cm_event *cm_event = NULL;
    int ret = -1;

    cm_event_channel = rdma_create_event_channel();
    if (!cm_event_channel) {
        rdma_error("Failed to create cm event channel, errno: %d\n", -errno);
        return -errno;
    }

    ret = rdma_create_id(cm_event_channel, &listener_id, NULL, RDMA_PS_TCP);
    if (ret) {
        rdma_error("Failed to create listener cm id, errno: %d\n", -errno);
        return -errno;
    }

    ret = rdma_bind_addr(listener_id, (struct sockaddr *)server_addr);
    if (ret) {
        rdma_error("Failed to bind to address, errno: %d\n", -errno);
        return -errno;
    }

    ret = rdma_listen(listener_id, 10);
    if (ret) {
        rdma_error("Failed to listen, errno: %d\n", -errno);
        return -errno;
    }

    printf("Server listening on port: %d\n", ntohs(server_addr->sin_port));
    debug("Waiting for client connection...\n");

    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_CONNECT_REQUEST, &cm_event);
    if (ret) {
        rdma_error("Failed to get connect request, ret: %d\n", ret);
        return ret;
    }

    cm_server_id = cm_event->id;
    ret = rdma_ack_cm_event(cm_event);
    if (ret) {
        rdma_error("Failed to ack cm event, errno: %d\n", -errno);
        return -errno;
    }

    return 0;
}

static int setup_client_resources() {
    int ret = -1;

    pd = ibv_alloc_pd(cm_server_id->verbs);
    if (!pd) {
        rdma_error("Failed to alloc pd, errno: %d\n", -errno);
        return -errno;
    }

    io_completion_channel = ibv_create_comp_channel(cm_server_id->verbs);
    if (!io_completion_channel) {
        rdma_error("Failed to create completion channel, errno: %d\n", -errno);
        return -errno;
    }

    server_cq = ibv_create_cq(cm_server_id->verbs, CQ_CAPACITY, NULL, io_completion_channel, 0);
    if (!server_cq) {
        rdma_error("Failed to create CQ, errno: %d\n", -errno);
        return -errno;
    }

    ret = ibv_req_notify_cq(server_cq, 0);
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
    qp_init_attr.recv_cq = server_cq;
    qp_init_attr.send_cq = server_cq;

    ret = rdma_create_qp(cm_server_id, pd, &qp_init_attr);
    if (ret) {
        rdma_error("Failed to create QP, errno: %d\n", -errno);
        return -errno;
    }
    server_qp = cm_server_id->qp;

    return 0;
}

static int accept_client_connection() {
    struct rdma_conn_param conn_param;
    struct rdma_cm_event *cm_event = NULL;
    int ret = -1;

    if (!cm_server_id || !server_qp) {
        rdma_error("Client resources not properly initialized\n");
        return -EINVAL;
    }

    client_metadata_mr = rdma_buffer_register(pd, &client_metadata_attr,
        sizeof(client_metadata_attr), IBV_ACCESS_LOCAL_WRITE);
    if (!client_metadata_mr) {
        rdma_error("Failed to register client metadata buffer\n");
        return -ENOMEM;
    }

    client_recv_sge.addr = (uint64_t)client_metadata_mr->addr;
    client_recv_sge.length = client_metadata_mr->length;
    client_recv_sge.lkey = client_metadata_mr->lkey;

    bzero(&client_recv_wr, sizeof(client_recv_wr));
    client_recv_wr.sg_list = &client_recv_sge;
    client_recv_wr.num_sge = 1;

    ret = ibv_post_recv(server_qp, &client_recv_wr, &bad_client_recv_wr);
    if (ret) {
        rdma_error("Failed to post recv for client metadata, errno: %d\n", ret);
        return ret;
    }

    bzero(&conn_param, sizeof(conn_param));
    conn_param.responder_resources = 1;
    conn_param.initiator_depth = 1;

    ret = rdma_accept(cm_server_id, &conn_param);
    if (ret) {
        rdma_error("Failed to accept connection, errno: %d\n", -errno);
        return -errno;
    }

    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_ESTABLISHED, &cm_event);
    if (ret) {
        rdma_error("Failed to get established event, ret: %d\n", ret);
        return ret;
    }

    ret = rdma_ack_cm_event(cm_event);
    if (ret) {
        rdma_error("Failed to ack established event, errno: %d\n", -errno);
        return -errno;
    }

    printf("Client connected successfully\n");
    return 0;
}

static int send_server_metadata_to_client() {
    struct ibv_wc wc;
    int ret = -1;

    server_buffer_mr = rdma_buffer_alloc(pd, SERVER_BUFFER_SIZE,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!server_buffer_mr) {
        rdma_error("Failed to allocate server data buffer\n");
        return -ENOMEM;
    }

    server_metadata_attr.address = (uint64_t)server_buffer_mr->addr;
    server_metadata_attr.length = server_buffer_mr->length;
    server_metadata_attr.stag.local_stag = server_buffer_mr->lkey;

    server_metadata_mr = rdma_buffer_register(pd, &server_metadata_attr,
        sizeof(server_metadata_attr), IBV_ACCESS_LOCAL_WRITE);
    if (!server_metadata_mr) {
        rdma_error("Failed to register server metadata\n");
        return -ENOMEM;
    }

    server_send_sge.addr = (uint64_t)server_metadata_mr->addr;
    server_send_sge.length = server_metadata_mr->length;
    server_send_sge.lkey = server_metadata_mr->lkey;

    bzero(&server_send_wr, sizeof(server_send_wr));
    server_send_wr.opcode = IBV_WR_SEND;
    server_send_wr.sg_list = &server_send_sge;
    server_send_wr.num_sge = 1;
    server_send_wr.send_flags = IBV_SEND_SIGNALED;

    ret = ibv_post_send(server_qp, &server_send_wr, &bad_server_send_wr);
    if (ret) {
        rdma_error("Failed to post send, errno: %d\n", ret);
        return ret;
    }

    ret = process_work_completion_events(io_completion_channel, &wc, 1);
    if (ret <= 0) {
        rdma_error("Failed to get send completion, ret: %d\n", ret);
        return ret;
    }

    show_rdma_buffer_attr(&server_metadata_attr);
    return 0;
}

static void process_client_operations() {
    struct ibv_wc wc;
    int ret;

    while (1) {
        ret = process_work_completion_events(io_completion_channel, &wc, 1);
        if (ret <= 0) break;

        if (wc.opcode == IBV_WR_RDMA_WRITE || wc.opcode == IBV_WR_RDMA_READ) {
            continue;
        }

        ret = ibv_post_recv(server_qp, &client_recv_wr, &bad_client_recv_wr);
        if (ret) break;
    }
}

static int disconnect_and_cleanup() {
    struct rdma_cm_event *cm_event = NULL;
    int ret = -1;

    process_client_operations();

    ret = rdma_disconnect(cm_server_id);
    if (ret) {
        rdma_error("Failed to disconnect, errno: %d\n", -errno);
        return -errno;
    }

    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_DISCONNECTED, &cm_event);
    if (ret) {
        rdma_error("Failed to get disconnect event, ret: %d\n", ret);
        return ret;
    }
    rdma_ack_cm_event(cm_event);

    if (server_buffer_mr) rdma_buffer_free(server_buffer_mr);
    if (server_metadata_mr) rdma_buffer_deregister(server_metadata_mr);
    if (client_metadata_mr) rdma_buffer_deregister(client_metadata_mr);
    if (server_qp) rdma_destroy_qp(cm_server_id);
    if (server_cq) ibv_destroy_cq(server_cq);
    if (io_completion_channel) ibv_destroy_comp_channel(io_completion_channel);
    if (pd) ibv_dealloc_pd(pd);
    if (cm_server_id) rdma_destroy_id(cm_server_id);
    if (listener_id) rdma_destroy_id(listener_id);
    if (cm_event_channel) rdma_destroy_event_channel(cm_event_channel);

    printf("Server resource cleanup complete\n");
    return 0;
}

int main(int argc, char **argv) {
    int ret, option;
    struct sockaddr_in server_sockaddr;
    bzero(&server_sockaddr, sizeof(server_sockaddr));
    server_sockaddr.sin_family = AF_INET;
    server_sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);

    while ((option = getopt(argc, argv, "p:")) != -1) {
        switch (option) {
            case 'p':
                server_sockaddr.sin_port = htons(strtol(optarg, NULL, 0));
                break;
            default:
                usage();
                break;
        }
    }

    if (!server_sockaddr.sin_port) {
        server_sockaddr.sin_port = htons(DEFAULT_RDMA_PORT);
    }

    ret = start_rdma_server(&server_sockaddr);
    if (ret) return ret;

    ret = setup_client_resources();
    if (ret) return ret;

    ret = accept_client_connection();
    if (ret) return ret;

    ret = send_server_metadata_to_client();
    if (ret) return ret;

    ret = disconnect_and_cleanup();
    return ret;
}