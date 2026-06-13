# Author: Animesh Trivedi
cmake_minimum_required(VERSION 3.5)
project(rdma-example)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)  # 或Debug

set(CMAKE_CXX_COMPILER g++) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Ofast -ffast-math -funroll-loops -march=native")

find_library(IBVERBS_LIBRARY ibverbs)
find_library(RDMACM_LIBRARY rdmacm)

link_libraries(pthread ${IBVERBS_LIBRARY} ${RDMACM_LIBRARY})

add_executable(rdma_server rdma_common.c rdma_server.c)
add_executable(rdma_client rdma_common.c rdma_client.c)