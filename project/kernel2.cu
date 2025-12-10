#include "common.h"
#include "timer.h"

/*
 * Building off Baseline.
 * One warp assigned to each row of A. Threads in a single warp collaborate to work on sections of a single row.
 * 
 * AB = C, where 
 * A, B, C are N-by-N,
 * A = B, and 
 * C_ij = sum_k{ A_ik * B_kj }
 */
__global__ void spmspm_kernel2(
    	const CSRMatrix* A, 
    	const CSRMatrix* B, 
    	COOMatrix* C, 
    	float* fullyConstructedCMatrix) {

    	unsigned int warpId = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32); // entire warp assigned to single row
    	unsigned int lane = threadIdx.x % 32; // which sections of row to work in
    	unsigned int N = A->numRows;

    	if (warpId >= N) return;

    	unsigned int i = warpId;  // which row to work on

    	float* fullyConstructedCRow = &fullyConstructedCMatrix[i * N];

    	unsigned int rowStart = A->rowPtrs[i];
    	unsigned int rowEnd = A->rowPtrs[i + 1];
    	unsigned int nnzA = rowEnd - rowStart;

    	for (unsigned int t = lane; t < nnzA; t += 32) {
    		unsigned int kPtr = rowStart + t;
    		unsigned int k = A->colIdxs[kPtr];

    		for (unsigned int jPtr = B->rowPtrs[k]; jPtr < B->rowPtrs[k + 1]; ++jPtr) {
        		unsigned int j = B->colIdxs[jPtr];
        		atomicAdd(&fullyConstructedCRow[j], A->values[kPtr] * B->values[jPtr]);
    		}
    	}
    	__syncthreads(); 

    	// Write nonzero dot products to COOMatrix C
    	// Assumes all matrix elements in A and B are 
    	// nonnegative (as is in the CPU implementation)
    	for (unsigned int j = lane; j < N; j += 32) {
    		float dotProduct = fullyConstructedCRow[j];
        	if (dotProduct != 0) {
            		unsigned int pos = atomicAdd(&C->numNonzeros, 1u);
            		C->rowIdxs[pos] = i;
            		C->colIdxs[pos] = j;
            		C->values[pos] = dotProduct;
        	}
    	}
}


void spmspm_gpu2(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
    	unsigned int N = csrMatrix1->numRows;
    	unsigned int threadsPerBlock = 128;
    	unsigned int warpsPerBlock = threadsPerBlock / 32;
    	unsigned int numBlocks = (N + warpsPerBlock - 1) / warpsPerBlock;
	float* fullyConstructedCMatrix_d;
	cudaMalloc(&fullyConstructedCMatrix_d, sizeof(float) * N * N);
	cudaMemset(fullyConstructedCMatrix_d, 0, sizeof(float) * N * N);
	spmspm_kernel2<<<numBlocks, threadsPerBlock>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d, fullyConstructedCMatrix_d);
}

