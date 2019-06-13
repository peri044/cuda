#include <math.h>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <iostream>

// Gauss Jordan elimination on CPU.
void GaussianEliminationCPU(float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot){

    // Initialize the output matrix with the input matrix as we will transform the input
	for (int i = 0; i<numberOfRows; i++){
		for (int j = 0; j<numberOfColumns; j++){
			outputMatrix[i][j] = matrix[i][j];
		}
	}
	
	for (int r = 0; r<numberOfRows; r++){
		float scaleFactor = outputMatrix[r][r];
		for (int c = 0; c<numberOfColumns; c++){
			outputMatrix[r][c] = (float)outputMatrix[r][c] / (float)scaleFactor;
        }

        for (int k = 0; k<numberOfRows; k++){
            float temp = outputMatrix[k][r];
            for (int c = 0; c<numberOfColumns; c++){
                if (k != r){
                    outputMatrix[k][c] = outputMatrix[k][c] - temp*outputMatrix[r][c];
                }
            }
        }

	}
}
