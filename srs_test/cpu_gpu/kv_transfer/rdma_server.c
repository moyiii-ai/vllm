/*
 * Modified RDMA server with GPU Direct RDMA support
 * Author: Animesh Trivedi
 *         atrivedi@apache.org
 * Modified to use GPU memory instead of CPU memory for RDMA operations
 * Added support for device ID parameter and dynamic port binding
 */

#include "rdma_common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <poll.h>
#include <getopt.h>
#include <cuda_device_runtime_api.h>


/* RDMA resources for GPU Direct */
static struct ibv_mr *gpu_buffer_mr = NULL;  // MR for GPU memory
static void *gpu_buffer = NULL;              // GPU memory buffer
static int gpu_device_id = -1;               // GPU device ID (-1 = CPU only)
static int use_gpu = 0;                      // Flag to use GPU or CPU
static const uint16_t BASE_PORT = 60000;     // Base port, actual port = BASE_PORT + device ID
static const size_t BUFFER_SIZE = (32ULL * 1024ULL * 1024ULL); // 128MB buffer
static void *cpu_buffer = NULL;              // CPU memory buffer for CPU-only mode

/* These are the RDMA resources needed to setup an RDMA connection */
static struct rdma_event_channel *cm_event_channel = NULL;
static struct rdma_cm_id *cm_server_id = NULL, *cm_client_id = NULL;
static struct ibv_pd *pd = NULL;
static struct ibv_comp_channel *io_completion_channel = NULL;
static struct ibv_cq *cq = NULL;
static struct ibv_qp_init_attr qp_init_attr;
static struct ibv_qp *client_qp = NULL;
static struct ibv_mr *client_metadata_mr = NULL, *server_metadata_mr = NULL;
static struct rdma_buffer_attr client_metadata_attr, server_metadata_attr;
static struct ibv_recv_wr client_recv_wr, *bad_client_recv_wr = NULL;
static struct ibv_send_wr server_send_wr, *bad_server_send_wr = NULL;
static struct ibv_sge client_recv_sge, server_send_sge;

/* Check if GPU device is valid */
static int is_valid_gpu_device(int dev_id) {
    int dev_count;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return (dev_id >= 0 && dev_id < dev_count) ? 1 : 0;
}

/* Parse command line arguments to get GPU device ID */
static int parse_args(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "d:c")) != -1) {
        switch (opt) {
            case 'd':
                gpu_device_id = atoi(optarg);
                use_gpu = 1;
                // Validate device ID
                if (!is_valid_gpu_device(gpu_device_id)) {
                    int dev_count;
                    cudaGetDeviceCount(&dev_count);
                    fprintf(stderr, "Error: Invalid device ID. Valid range is 0-%d\n", dev_count - 1);
                    return -1;
                }
                break;
            case 'c':
                use_gpu = 0;
                gpu_device_id = -1;
                break;
            default:
                fprintf(stderr, "Usage: %s [-d <device_id>] [-c]\n", argv[0]);
                fprintf(stderr, "  -d: GPU device ID (default: use GPU 0)\n");
                fprintf(stderr, "  -c: CPU-only mode (no GPU)\n");
                return -1;
        }
    }
    return 0;
}

/* Get port based on GPU device ID */
static uint16_t get_port_by_device_id() {
    if (use_gpu) {
        return BASE_PORT + gpu_device_id;
    } else {
        // For CPU-only mode, use base port
        return BASE_PORT;
    }
}

/* Setup client resources (PD, CQ, QP) */
static int setup_client_resources()
{
    int ret = -1;
    if(!cm_client_id){
        rdma_error("Client id is still NULL \n");
        return -EINVAL;
    }

    pd = ibv_alloc_pd(cm_client_id->verbs);
    if (!pd) {
        rdma_error("Failed to allocate a protection domain errno: %d\n", -errno);
        return -errno;
    }
    debug("A new protection domain is allocated at %p \n", pd);

    io_completion_channel = ibv_create_comp_channel(cm_client_id->verbs);
    if (!io_completion_channel) {
        rdma_error("Failed to create an I/O completion event channel, %d\n", -errno);
        return -errno;
    }
    debug("An I/O completion event channel is created at %p \n", io_completion_channel);

    cq = ibv_create_cq(cm_client_id->verbs, CQ_CAPACITY, NULL, io_completion_channel, 0);
    if (!cq) {
        rdma_error("Failed to create a completion queue (cq), errno: %d\n", -errno);
        return -errno;
    }
    debug("Completion queue (CQ) is created at %p with %d elements \n", cq, cq->cqe);

    ret = ibv_req_notify_cq(cq, 0);
    if (ret) {
        rdma_error("Failed to request notifications on CQ errno: %d \n", -errno);
        return -errno;
    }

    bzero(&qp_init_attr, sizeof qp_init_attr);
    qp_init_attr.cap.max_recv_sge = MAX_SGE;
    qp_init_attr.cap.max_recv_wr = MAX_WR;
    qp_init_attr.cap.max_send_sge = MAX_SGE;
    qp_init_attr.cap.max_send_wr = MAX_WR;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.send_cq = cq;

    ret = rdma_create_qp(cm_client_id, pd, &qp_init_attr);
    if (ret) {
        rdma_error("Failed to create QP due to errno: %d\n", -errno);
        return -errno;
    }
    client_qp = cm_client_id->qp;
    debug("Client QP created at %p\n", client_qp);
    return ret;
}

/* Initialize RDMA buffer (CPU/GPU) and register for remote access */
static int setup_rdma_buffer(size_t size)
{
    cudaError_t cuda_ret;

    if (!use_gpu) {
        // CPU-only mode: allocate CPU memory
        cpu_buffer = malloc(size);
        if (!cpu_buffer) {
            rdma_error("malloc for CPU buffer failed\n");
            return -1;
        }
        printf("CPU buffer address: %p (size: %zu bytes)\n", cpu_buffer, size);
        
        // Register CPU memory with RDMA
        gpu_buffer_mr = ibv_reg_mr(pd, cpu_buffer, size,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!gpu_buffer_mr) {
            rdma_error("ibv_reg_mr for CPU buffer failed: %d\n", -errno);
            free(cpu_buffer);
            return -1;
        }
        printf("Registered CPU buffer as MR (lkey: 0x%x, rkey: 0x%x)\n",
            gpu_buffer_mr->lkey, gpu_buffer_mr->rkey);
        
        return 0;
    }

    // GPU mode: allocate GPU memory
    // Set and check GPU device
    cudaError_t err = cudaSetDevice(gpu_device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", gpu_device_id, cudaGetErrorString(err));
        return -1;
    }
    printf("Using GPU device %d\n", gpu_device_id);

    // Allocate GPU memory
    cuda_ret = cudaMalloc(&gpu_buffer, size);
    if (cuda_ret != cudaSuccess) {
        rdma_error("cudaMalloc failed: %s\n", cudaGetErrorString(cuda_ret));
        return -1;
    }
    printf("GPU buffer address: %p (size: %zu bytes)\n", gpu_buffer, size);
    debug("Allocated GPU buffer at %p (size: %zu)\n", gpu_buffer, size);

    // Register GPU memory with RDMA
    gpu_buffer_mr = ibv_reg_mr(pd, gpu_buffer, size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!gpu_buffer_mr) {
        rdma_error("ibv_reg_mr for GPU buffer failed: %d\n", -errno);
        cudaFree(gpu_buffer);
        return -1;
    }
    printf("Registered GPU buffer as MR (lkey: 0x%x, rkey: 0x%x)\n",
        gpu_buffer_mr->lkey, gpu_buffer_mr->rkey);

    return 0;
}

/* Start RDMA server and wait for connection request */
static int start_rdma_server(struct sockaddr_in *server_addr)
{
    struct rdma_cm_event *cm_event = NULL;
    int ret = -1;
    cm_event_channel = rdma_create_event_channel();
    if (!cm_event_channel) {
        rdma_error("Creating cm event channel failed with errno : (%d)", -errno);
        return -errno;
    }
    debug("RDMA CM event channel is created successfully at %p \n", cm_event_channel);

    ret = rdma_create_id(cm_event_channel, &cm_server_id, NULL, RDMA_PS_TCP);
    if (ret) {
        rdma_error("Creating server cm id failed with errno: %d ", -errno);
        return -errno;
    }
    debug("A RDMA connection id for the server is created \n");

    ret = rdma_bind_addr(cm_server_id, (struct sockaddr*) server_addr);
    if (ret) {
        rdma_error("Failed to bind server address, errno: %d \n", -errno);
        return -errno;
    }
    debug("Server RDMA CM id is successfully binded \n");

    ret = rdma_listen(cm_server_id, 8);
    if (ret) {
        rdma_error("rdma_listen failed to listen on server address, errno: %d ", -errno);
        return -errno;
    }
    if (use_gpu) {
        printf("Server (GPU %d) is listening successfully at: %s , port: %d \n",
            gpu_device_id, inet_ntoa(server_addr->sin_addr), ntohs(server_addr->sin_port));
    } else {
        printf("Server (CPU-only mode) is listening successfully at: %s , port: %d \n",
            inet_ntoa(server_addr->sin_addr), ntohs(server_addr->sin_port));
    }

    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_CONNECT_REQUEST, &cm_event);
    if (ret) {
        rdma_error("Failed to get cm event, ret = %d \n" , ret);
        return ret;
    }

    cm_client_id = cm_event->id;
    ret = rdma_ack_cm_event(cm_event);
    if (ret) {
        rdma_error("Failed to acknowledge the cm event errno: %d \n", -errno);
        return ret;
    }
    debug("A new RDMA client connection id is stored at %p\n", cm_client_id);
    return ret;
}

/* Accept client connection and exchange metadata */
static int accept_client_connection()
{
    struct rdma_conn_param conn_param;
    struct rdma_cm_event *cm_event = NULL;
    struct ibv_wc wc;
    int ret = -1;

    if(!cm_client_id || !client_qp) {
        rdma_error("Client resources are not properly setup\n");
        return -EINVAL;
    }

    // Prepare receive buffer for client metadata
    client_metadata_mr = rdma_buffer_register(pd, &client_metadata_attr,
        sizeof(client_metadata_attr), IBV_ACCESS_LOCAL_WRITE);
    if(!client_metadata_mr) {
        rdma_error("Failed to register client attr buffer\n");
        return -ENOMEM;
    }

    client_recv_sge.addr = (uint64_t) client_metadata_mr->addr;
    client_recv_sge.length = client_metadata_mr->length;
    client_recv_sge.lkey = client_metadata_mr->lkey;

    bzero(&client_recv_wr, sizeof(client_recv_wr));
    client_recv_wr.sg_list = &client_recv_sge;
    client_recv_wr.num_sge = 1;

    ret = ibv_post_recv(client_qp, &client_recv_wr, &bad_client_recv_wr);
    if (ret) {
        rdma_error("Failed to pre-post the receive buffer, errno: %d \n", ret);
        return ret;
    }
    debug("Receive buffer pre-posting is successful \n");

    // Accept connection
    bzero(&conn_param, sizeof(conn_param));
    conn_param.responder_resources = 3; 
    conn_param.initiator_depth = 3;
    ret = rdma_accept(cm_client_id, &conn_param);
    if (ret) {
        rdma_error("Failed to accept the client connection, errno: %d \n", -errno);
        return ret;
    }

    // Wait for connection established event
    ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_ESTABLISHED, &cm_event);
    if (ret) {
        rdma_error("Failed to get established event, ret = %d \n", ret);
        return ret;
    }
    ret = rdma_ack_cm_event(cm_event);
    if (ret) {
        rdma_error("Failed to ack established event, errno: %d \n", -errno);
        return ret;
    }
    debug("Client connection established \n");

    // Wait for client metadata
    ret = process_work_completion_events(io_completion_channel, &wc, 1);
    if (ret <= 0) {
        rdma_error("Failed to get client metadata, ret = %d \n", ret);
        return ret;
    }
    debug("Received client metadata \n");

    // Allocate/register RDMA buffer after we know required size (like ib_*_bw)
    size_t buf_size = (size_t)client_metadata_attr.length;
    if (buf_size == 0) buf_size = BUFFER_SIZE;
    if (setup_rdma_buffer(buf_size) != 0) {
        rdma_error("Failed to setup RDMA buffer (size=%zu)\n", buf_size);
        return -1;
    }

    // Prepare server metadata with RDMA buffer info
    // In CPU-only mode, gpu_buffer is NULL; we must advertise cpu_buffer instead.
    void *advertised_buf = use_gpu ? gpu_buffer : cpu_buffer;
    if (!advertised_buf) {
        rdma_error("Server buffer is NULL (use_gpu=%d)\n", use_gpu);
        return -EINVAL;
    }
    server_metadata_attr.address = (uint64_t)advertised_buf;
    server_metadata_attr.length = gpu_buffer_mr->length;
    // Peer must use rkey for remote access
    server_metadata_attr.stag.local_stag = gpu_buffer_mr->rkey;

    // Send server metadata to client
    server_metadata_mr = rdma_buffer_register(pd, &server_metadata_attr,
        sizeof(server_metadata_attr), IBV_ACCESS_LOCAL_WRITE);
    if (!server_metadata_mr) {
        rdma_error("Failed to register server metadata buffer\n");
        return -ENOMEM;
    }

    server_send_sge.addr = (uint64_t) server_metadata_mr->addr;
    server_send_sge.length = server_metadata_mr->length;
    server_send_sge.lkey = server_metadata_mr->lkey;

    bzero(&server_send_wr, sizeof(server_send_wr));
    server_send_wr.opcode = IBV_WR_SEND;
    server_send_wr.sg_list = &server_send_sge;
    server_send_wr.num_sge = 1;
    server_send_wr.send_flags = IBV_SEND_SIGNALED;

    ret = ibv_post_send(client_qp, &server_send_wr, &bad_server_send_wr);
    if (ret) {
        rdma_error("Failed to post send for server metadata, errno: %d \n", ret);
        return ret;
    }

    // Wait for send completion
    ret = process_work_completion_events(io_completion_channel, &wc, 1);
    if (ret <= 0) {
        rdma_error("Failed to complete metadata send, ret = %d \n", ret);
        return ret;
    }
    debug("Server metadata sent to client \n");

    return 0;
}

/* Cleanup resources */
static void cleanup_resources()
{
    // Cleanup GPU resources first
    if (gpu_buffer_mr) {
        ibv_dereg_mr(gpu_buffer_mr);
        gpu_buffer_mr = NULL;
    }
    if (use_gpu) {
        if (gpu_buffer) {
            cudaFree(gpu_buffer);
            gpu_buffer = NULL;
        }
    } else {
        if (cpu_buffer) {
            free(cpu_buffer);
            cpu_buffer = NULL;
        }
    }

    // Cleanup other RDMA resources
    if (client_metadata_mr) {
        rdma_buffer_deregister(client_metadata_mr);
        client_metadata_mr = NULL;
    }
    if (server_metadata_mr) {
        rdma_buffer_deregister(server_metadata_mr);
        server_metadata_mr = NULL;
    }
    if (client_qp) {
        rdma_destroy_qp(cm_client_id);
        client_qp = NULL;
    }
    if (cq) {
        ibv_destroy_cq(cq);
        cq = NULL;
    }
    if (io_completion_channel) {
        ibv_destroy_comp_channel(io_completion_channel);
        io_completion_channel = NULL;
    }
    if (pd) {
        ibv_dealloc_pd(pd);
        pd = NULL;
    }
    if (cm_client_id) {
        rdma_destroy_id(cm_client_id);
        cm_client_id = NULL;
    }
    if (cm_server_id) {
        rdma_destroy_id(cm_server_id);
        cm_server_id = NULL;
    }
    if (cm_event_channel) {
        rdma_destroy_event_channel(cm_event_channel);
        cm_event_channel = NULL;
    }
}

static void process_client_operations()
{
    struct rdma_cm_event *cm_event = NULL;
    int ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_DISCONNECTED, &cm_event);
    if (ret) {
        rdma_error("Failed to get disconnect event, ret = %d\n", ret);
        return;
    }
    rdma_ack_cm_event(cm_event);
    printf("Client disconnected\n");
}

int main(int argc, char **argv)
{
    struct sockaddr_in server_addr;
    int ret = 0;
    // Parse command line arguments
    ret = parse_args(argc, argv);
    if (ret != 0) {
        return ret;
    }
    // Initialize server address with dynamic port
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(get_port_by_device_id());

    // Initialize RDMA server
    ret = start_rdma_server(&server_addr);
    if (ret) {
        rdma_error("Failed to start RDMA server, ret = %d\n", ret);
        cleanup_resources();
        return ret;
    }


    // Setup client resources
    ret = setup_client_resources();
    if (ret) {
        rdma_error("Failed to setup client resources, ret = %d\n", ret);
        cleanup_resources();
        return ret;
    }

    // Accept connection and exchange metadata
    ret = accept_client_connection();
    if (ret) {
        rdma_error("Failed to accept client connection, ret = %d\n", ret);
        cleanup_resources();
        return ret;
    }

    process_client_operations();

    // Cleanup
    cleanup_resources();
    if (use_gpu) {
        printf("Server (GPU %d) cleanup complete\n", gpu_device_id);
    } else {
        printf("Server (CPU-only mode) cleanup complete\n");
    }
    return ret;
}