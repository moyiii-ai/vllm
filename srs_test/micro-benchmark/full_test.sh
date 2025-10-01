./run_throughput_cuda.sh 1 read 0 > one_read_cuda.dat
./run_throughput_cuda.sh 1 write 0 > one_write_cuda.dat
./run_throughput_cuda.sh 2 read > two_read_cuda.dat
./run_throughput_cuda.sh 2 write > two_write_cuda.dat
./run_throughput_global.sh 1 read 0 > one_read_global.dat
./run_throughput_global.sh 1 write 0 > one_write_global.dat
./run_throughput_global.sh 2 read > two_read_global.dat
./run_throughput_global.sh 2 write > two_write_global.dat
./run_throughput_copy.sh 1 read 0 > one_read_copy.dat
./run_throughput_copy.sh 1 write 0 > one_write_copy.dat
./run_throughput_copy.sh 2 read > two_read_copy.dat
./run_throughput_copy.sh 2 write > two_write_copy.dat