# Gauss Jordan elimination on GPU

## Compile the CUDA kernel
nvcc -std=c++11 -c gpu.cu
g++ -std=c++11 main.cpp cpu.cpp gpu.o -I /usr/local/cuda-10.0/include -lcudart -L /usr/local/cuda-10.0/lib64 -o gauss
