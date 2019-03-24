#include <math.h>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <iostream>

void GaussianEliminationCPU(float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot){

	for (int i = 0; i<numberOfRows; i++){
		for (int j = 0; j<numberOfColumns; j++){
			outputMatrix[i][j] = matrix[i][j];
			//printf("%f \t", outputMatrix[i][j]);
		}
		//printf("\n");
	}
	////Partial Pivoting
	//if(partialPivot){
	//
	//	float max = outputMatrix[0][0];
	//	int maxIndex;
	//	for (int x = 0; x < numberOfRows; x++){
	//		if (outputMatrix[x][0] > max){
	//			max = outputMatrix[x][0];
	//			maxIndex = x;
	//		}
	//	}

	//	float *currentRow = outputMatrix[r]; //Store the current row array pointer in a temporary variable
	//	outputMatrix[r] = outputMatrix[maxIndex];
	//	outputMatrix[maxIndex] = currentRow;//Swapping
	//}


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
		//Printing the output
	  //  printf("Final CPU output is \n");
		// for (int i = 0; i < numberOfRows; i++){
		// 	for (int j = 0; j < numberOfColumns; j++){
		// 		printf("%f \t", outputMatrix[i][j]);
		// 	}
		// 	printf("\n\n");
		// }

}
