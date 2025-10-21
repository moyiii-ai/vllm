cmake_minimum_required(VERSION 3.18)
project(rdma-example LANGUAGES C CUDA)

# Output directory configuration
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# C standard configuration
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# Explicitly specify CUDA architecture (resolves CMP0104 warning)
set(CMAKE_CUDA_ARCHITECTURES 89)

# CUDA compiler configuration
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# Find dependent libraries
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
find_library(RDMACM_LIBRARY rdmacm REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# Link libraries
link_libraries(
    pthread 
    ${IBVERBS_LIBRARY} 
    ${RDMACM_LIBRARY}
)

# Build rdma_server with CUDA support
add_executable(rdma_server 
    rdma_common.c 
    rdma_server.c
)
set_source_files_properties(rdma_server.c PROPERTIES LANGUAGE CUDA)
set_property(TARGET rdma_server PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(rdma_server PRIVATE cudart)

# Build rdma_client as standard C executable
add_executable(rdma_client 
    rdma_common.c 
    rdma_client.c
)
set_property(TARGET rdma_client PROPERTY LINKER_LANGUAGE C)