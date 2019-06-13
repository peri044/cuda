#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <string.h>

#include "Gaussian.h"

const int TILE_SIZE = 32;

bool checkForError(cudaError_t error, char* funcName);

// Scale kernel takes in each row and divides all other elements in row with the index element. For eg: in the first row index element is (1,1)...2nd row it is (2,2)...so on..
// It divides all other elements in the row with their corresponding index elements
__global__ void ScaleRowKernel(float* in, float* out, unsigned int numRows, unsigned int numCols, int pivot){

	int col = threadIdx.x + blockDim.x*blockIdx.x;
	int row = threadIdx.y + blockDim.y*blockIdx.y;

	if (row < numRows && col < numCols){

		if (row == pivot && col != pivot) out[row*numCols + col] = in[row*numCols + col] / in[pivot*numCols + pivot];  // The index (pivot) element for scaling ..I'm reading it from the input matrix !!

		if (row == pivot && col == pivot){ out[pivot*numCols + pivot] = 1; }  // Instead of dividing the index element by itself in the row, I simply make it 1 because it has to be....
		//else out[row*numCols + col] = in[row*numCols + col];
	}
}


// Subtraction kernel takes in each row....  multiplies the multiplying factor with the elements of pivot row and subtracts the elements of other rows with the pivot row
// Similar implementation to CPU
__global__ void SubtractionKernel(float* in, float* out, unsigned int numRows, unsigned int numCols, int pivot){

	int col = threadIdx.x + blockDim.x*blockIdx.x;
	int row = threadIdx.y + blockDim.y*blockIdx.y;

	if (row < numRows && col < numCols){
		if (row != pivot){
			out[row*numCols + col] = in[row*numCols + col] - (in[row*numCols + pivot] * in[pivot*numCols + col]);
		}
		else out[row*numCols + col] = in[pivot*numCols + col];
	}
}

bool GaussianEliminationGPU(float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot){


	cudaError_t status;
	bool error;
	int bytes = numberOfRows * numberOfColumns * sizeof(float);
	float scaleFactor = 0;
	float* out;
	float* in;

	// Initializing the values
	cudaMalloc((void**)&out, bytes);
	cudaMalloc((void**)&in, bytes);

	status = cudaGetLastError();
	error = checkForError(status, "At cudaMalloc");
	if (!error){
		cudaFree(out);
		cudaFree(in);
		return false;
	}

	//transfer of data from host to device
	for (int r = 0; r<numberOfRows; r++){
		cudaMemcpy(&out[r*numberOfColumns], matrix[r], numberOfColumns*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&in[r*numberOfColumns], matrix[r], numberOfColumns*sizeof(float), cudaMemcpyHostToDevice);
	}
	status = cudaGetLastError();
	error = checkForError(status, "At cudaMemcpy");
	if (!error){
		cudaFree(out);
		cudaFree(in);
		return false;
	}

	// Defining Grid and block
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);   // 2D block
	status = cudaGetLastError();
	error = checkForError(status, "At dimBlock");
	if (!error){
		cudaFree(out);
		cudaFree(in);
		return false;
	}

	//Using 2 Dimensional grid
	int gridSize1 = (int)ceil((((float)numberOfColumns) / (TILE_SIZE)));
	int gridSize2 = (int)ceil((((float)numberOfRows) / (TILE_SIZE)));
	dim3 dimGrid(gridSize1, gridSize2);
	status = cudaGetLastError();
	error = checkForError(status, "At dimGrid");
	if (!error){
		cudaFree(out);
		cudaFree(in);
		return false;
	}

	//Scaling each row and performing elementary row wise operations
	for (int p = 0; p<numberOfRows; p++){ // p is the pivot row

		ScaleRowKernel << < dimGrid, dimBlock >> >(in, out, numberOfRows, numberOfColumns, p);
		cudaDeviceSynchronize();
		cudaThreadSynchronize();
		status = cudaGetLastError();
		error = checkForError(status, "At ScaleRowKernel");
		if (!error){
			cudaFree(out);
			cudaFree(in);
			return false;
		}
		// After each operation on a row, I'm copying the output matrix (out) into input matrix (in)..so that next scaling row will have updated input matrix
		cudaMemcpy(in, out, numberOfRows*numberOfColumns*sizeof(float), cudaMemcpyDeviceToDevice);


		SubtractionKernel << < dimGrid, dimBlock >> >(in, out, numberOfRows, numberOfColumns, p);
		cudaDeviceSynchronize();
		cudaThreadSynchronize();
		status = cudaGetLastError();
		error = checkForError(status, "At SubtractionKernel");
		if (!error){
			printf("Error at Subtraction kernel row number : %d", p);
			cudaFree(out);
			cudaFree(in);
			return false;
		}
		// After each operation on a row, I'm copying the output matrix (out) into input matrix (in)..so that next scaling row will have updated input matrix
		cudaMemcpy(in, out, numberOfRows*numberOfColumns*sizeof(float), cudaMemcpyDeviceToDevice);

		cudaDeviceSynchronize();
		cudaThreadSynchronize();
	}

	for (int r = 0; r<numberOfRows; r++){
		cudaMemcpy(outputMatrix[r], &out[r*numberOfColumns],
			numberOfColumns*sizeof(float), cudaMemcpyDeviceToHost);
	}
	
	status = cudaGetLastError();
	error = checkForError(status, "After cudaMemcopy to host");
	if (!error){
		cudaFree(out);
		cudaFree(in);
		return false;
	}


	cudaFree(out);
	cudaFree(in);
	return true;
}

bool checkForError(cudaError_t error, char* funcName){
	bool success;
	if (error != cudaSuccess){
		success = false;
		printf("Device error at %s \n", funcName);
		printf("%s", cudaGetErrorString);
	}
	else{
		success = true;
	}
	return success;
}
