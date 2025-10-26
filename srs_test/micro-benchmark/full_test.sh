#!/bin/bash

./run_throughput.sh cuda 1 read 0 > one_read_cuda.dat
echo "Finished: one_read_cuda.dat"

./run_throughput.sh cuda 1 write 0 > one_write_cuda.dat
echo "Finished: one_write_cuda.dat"

./run_throughput.sh cuda 2 read > two_read_cuda.dat
echo "Finished: two_read_cuda.dat"

./run_throughput.sh cuda 2 write > two_write_cuda.dat
echo "Finished: two_write_cuda.dat"

./run_throughput.sh global 1 read 0 > one_read_global.dat
echo "Finished: one_read_global.dat"

./run_throughput.sh global 1 write 0 > one_write_global.dat
echo "Finished: one_write_global.dat"

./run_throughput.sh global 2 read > two_read_global.dat
echo "Finished: two_read_global.dat"

./run_throughput.sh global 2 write > two_write_global.dat
echo "Finished: two_write_global.dat"

./run_throughput.sh copy 1 read 0 > one_read_copy.dat
echo "Finished: one_read_copy.dat"

./run_throughput.sh copy 1 write 0 > one_write_copy.dat
echo "Finished: one_write_copy.dat"

./run_throughput.sh copy 2 read > two_read_copy.dat
echo "Finished: two_read_copy.dat"

./run_throughput.sh copy 2 write > two_write_copy.dat
echo "Finished: two_write_copy.dat"
