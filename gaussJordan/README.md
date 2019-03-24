# Gauss Jordan elimination on GPU
This repo has the code for gaussian elimination for solving linear systems on CUDA. Both CPU and GPU implementations are written and compared. 

## Compile the CUDA kernel
```
nvcc -std=c++11 -c gpu.cu
```
Link it to the executable and run
```
g++ -std=c++11 main.cpp cpu.cpp gpu.o -I /usr/local/cuda-10.0/include -lcudart -L /usr/local/cuda-10.0/lib64 -o gauss

./gauss.out
```
