# Gauss Jordan elimination on GPU
This repo has the code for <a href="https://en.wikipedia.org/wiki/Gaussian_elimination"> Gauss Jordan elimination</a> for solving a system of Linear equations on CUDA. 
Both CPU and GPU implementations are written and compared. 
![Alt text](gaussjordan.PNG?raw=true)

## Compile the CUDA kernel
```
nvcc -std=c++11 -c gpu.cu
```
Build the executable 
```
g++ -std=c++11 main.cpp cpu.cpp gpu.o -I /usr/local/cuda-10.0/include -lcudart -L /usr/local/cuda-10.0/lib64 -o gauss
```

Run the executable
```
./gauss.out --num_rows 32
```

* Add `--debug` to print the values of CPU and GPU outputs to inspect.

## Speed up table on RTX 2080Ti for different input matrix sizes.
| Input Size  | CPU time(ms) | GPU time (ms) | Speedup |
| ------------- | ------------- |------------- | ------------- |
| 3x4  | 0.004  |  0.507  | 0.007 |
| 4x5  | 0.005  |  0.569  | 0.008 |
| 8x9  | 0.01  |  0.712  | 0.014 |
| 16x17  | 0.05  |  1.041  | 0.048 |
| 32x33  | 0.369  | 1.742   | 0.21 |
| 64x65  | 2.868  |  3.022 | 0.94 |
| 128x129  | 17.028  | 5.719  | 2.97 |
| 256x257  | 65.832 |  11.10  | 5.92 |
| 512x513  | 419.384  |  25.96  | 16.1 |
| 1024x1025  | 3247.38  |  98.63  | 32.9 |
| 2048x2049  | 26284.2  |  504.42  | 52.1 |