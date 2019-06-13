#include <ctime> // clock()
#include <cmath> // sqrt()
#include <iostream> // cout, stream
#include <cuda.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "Gaussian.h"

const int ITERS = 1;

int main(int argc, char **argv){
    
    bool debug = false;
     
    // 2nd argument is number of rows
    int NUMROWS = atoi(argv[2]);
    int NUMCOLS = NUMROWS + 1;
    
    const char* debug_arg = "--debug";
    for(int i=0; i< argc; i++){
        if (strcmp(debug_arg, argv[i])==0)
        debug = true;
    }
    

	clock_t start, end;
	float tcpu, tgpu;
    // Define and initialize input and output matrices for CPU and GPU 
	float** input = new float*[NUMROWS]; 
	float** cpu_output = new float*[NUMROWS]; 
	float** gpu_output = new float*[NUMROWS];
     
	for (int i = 0; i<NUMROWS; i++){
		input[i] = new float[NUMCOLS];
		cpu_output[i] = new float[NUMCOLS];
		gpu_output[i] = new float[NUMCOLS];
		for (int j = 0; j<NUMCOLS; j++){
			input[i][j] = (float)rand() / ((float)RAND_MAX);
		}
	}

	std::cout << "Operating on a " << NUMROWS << " x " << NUMCOLS << " input matrix" << std::endl;
	float L2norm = 0;
	float sum = 0, delta = 0;
	bool success;
	bool partialPivot = false; //default is false. false = no pivoting, true = partial pivoting
    
    // Profile on CPU
	start = clock();
	for (int i = 0; i<ITERS; i++){
		GaussianEliminationCPU(input, NUMROWS, NUMCOLS, cpu_output, partialPivot);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	
    
    // Warm up pass on GPU
	success = GaussianEliminationGPU(input, NUMROWS, NUMCOLS, gpu_output, partialPivot);
	if (!success) {
		return 1;
	}
    // Profile on GPU
	start = clock();
	for (int i = 0; i<ITERS; i++){
		GaussianEliminationGPU(input, NUMROWS, NUMCOLS, gpu_output, partialPivot);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
    
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "CPU Result took " << tcpu << " ms"<< std::endl;
	std::cout << "GPU Result (direct) took " << tgpu << " ms" << std::endl;
    std::cout << "Speed up " << tcpu / tgpu << std::endl;
    
    // Calculate the L2 error between CPU and GPU output
	for (int r = 0; r<NUMROWS; r++){
		for (int c = 0; c<NUMCOLS; c++){
			delta += (gpu_output[r][c] - cpu_output[r][c]) * (gpu_output[r][c] - cpu_output[r][c]);
			sum += (gpu_output[r][c] * cpu_output[r][c]);
		}
	}

	L2norm = sqrt(delta / sum);
	std::cout << "Error: " << L2norm << std::endl << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    if (debug){
        
        int iter = 10;
        if (iter>NUMROWS) iter = NUMROWS;
        std::cout << " First " << iter << " output values of CPU:" << std::endl;
        for(int i=0; i<iter; i++)
            std::cout << cpu_output[i][NUMCOLS-1] << " ";
        std::cout << "\n --------------------------------------------------------" << std::endl;
        std::cout << " First " << iter << " output values of GPU:" << std::endl;
        for(int i=0; i<iter; i++)
            std::cout << gpu_output[i][NUMCOLS-1] << " ";
    }
    std::cout << "\nFinished Gauss Jordan elimination" << std::endl;
    // Release the memory
	for (int i = 0; i<NUMROWS; i++){
		delete[] input[i];
		delete[] cpu_output[i];
		delete[] gpu_output[i];
	}
	delete[] input;
	delete[] cpu_output;
	delete[] gpu_output;

	return 0;
}
