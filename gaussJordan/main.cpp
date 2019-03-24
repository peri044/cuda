#include <cstdlib> // malloc(), free()
#include <ctime> // time(), clock()
#include <cmath> // sqrt()
#include <iostream> // cout, stream
#include <math.h>
#include <string>
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "Gaussian.h"

const unsigned int NUMROWS = 1024;
const unsigned int NUMCOLS = 1024;
const int ITERS = 1;


int main(){

	clock_t start, end;
	float tcpu, tgpu;
	float** m = new float*[NUMROWS]; //input matrix
	float** om = new float*[NUMROWS]; //outputmatrix
	float** ppom = new float*[NUMROWS];
	float** gpuom = new float*[NUMROWS];
	//initialize m with values and allocate space for om
	//std::cout<<"Input matrix 'm' is: "<<std::endl;
	for (int i = 0; i<NUMROWS; i++){
		m[i] = new float[NUMCOLS];
		om[i] = new float[NUMCOLS];
		ppom[i] = new float[NUMCOLS];
		gpuom[i] = new float[NUMCOLS];
		for (int j = 0; j<NUMCOLS; j++){
			m[i][j] = (float)rand() / ((float)RAND_MAX);
		}
		//std::cout<< std::endl;
	}
	//std::cin.get();
	std::cout << "Operating on a " << NUMROWS << " x " << NUMCOLS << " matrix" << std::endl;
	float L2norm = 0;
	float sum = 0, delta = 0;
	bool success;
	bool partialPivot = true; //default is false. false = no pivoting, true = partial pivoting
		start = clock();
	for (int i = 0; i<ITERS; i++){
		GaussianEliminationCPU(m, NUMROWS, NUMCOLS, ppom, partialPivot);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "Host Result (partial Pivoting) took " << tcpu << " ms"
		<< std::endl;
	std::cout << std::endl;

	// partialPivot = false;
	// start = clock();
	// for (int i = 0; i<ITERS; i++){
	// 	GaussianEliminationCPU(m, NUMROWS, NUMCOLS, om, partialPivot);
	// }
	// end = clock();
	// tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// // Display the result
	// std::cout << "Host Result (direct) took " << tcpu << " ms" << std::endl;
	// for (int r = 0; r<NUMROWS; r++){
	// 	for (int c = 0; c<NUMCOLS; c++){
	// 		delta += (ppom[r][c] - om[r][c]) * (ppom[r][c] - om[r][c]);
	// 		sum += (ppom[r][c] * om[r][c]);
	// 	}
	// }
	//
	// L2norm = sqrt(delta / sum);
	// std::cout << "Error: " << L2norm << std::endl << std::endl;

	success = GaussianEliminationGPU(m, NUMROWS, NUMCOLS, gpuom, partialPivot);
	if (!success) {
		std::cout << "\nGaussian Elimination GPU: * Device error! * \n" << std::endl;
		std::cin.get();
		return 1;
	}

	start = clock();
	for (int i = 0; i<ITERS; i++){
		GaussianEliminationGPU(m, NUMROWS, NUMCOLS, gpuom, partialPivot);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "Device Result (direct) took " << tgpu << " ms" << std::endl;
	for (int r = 0; r<NUMROWS; r++){
		for (int c = 0; c<NUMCOLS; c++){
			//std::cout<<"gpuom[r][c]"<<gpuom[r][c]<<std::endl;
			//std::cout<<"om[r][c] "<<om[r][c]<<std::endl;
			delta += (gpuom[r][c] - ppom[r][c]) * (gpuom[r][c] - ppom[r][c]);
			sum += (gpuom[r][c] * ppom[r][c]);
		}
	}
	/*
	std::cout<<"delta of gpu = "<<delta<<std::endl;
	std::cout<<"Output matrix 'gpuom' is: "<<std::endl;
	for(int i=0; i<NUMROWS;i++){
	for(int j=0; j<NUMCOLS;j++){
	std::cout<<gpuom[i][j] <<"\t";
	}
	std::cout<< std::endl;
	}
	*/
	L2norm = sqrt(delta / sum);
	std::cout << "Error: " << L2norm << std::endl << std::endl;
	//std::cout<<"done with GPU!"<<std::endl;
	std::cin.get();
	for (int i = 0; i<NUMROWS; i++){
		delete[] om[i];
		delete[] m[i];
		delete[] ppom[i];
		delete[] gpuom[i];
	}
	delete[] m;
	delete[] om;
	delete[] ppom;
	delete[] gpuom;

	return 0;
}
