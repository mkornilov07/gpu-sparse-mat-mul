#include "common.h"
#include "timer.h"

/*
 * Parallel implementation of SpMSpM with 
 * one thread assigned to each A_iq.
 * 
 * AB = C, where 
 * A, B, C are N-by-N,
 * A = B, and 
 * C_ij = sum_k{ A_ik * B_kj }
 */
__global__ void spmspm_kernel1(
                const CSRMatrix* A,
                const CSRMatrix* B,
                float* fullyConstructedCMatrix) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int q = blockIdx.x * blockDim.x + threadIdx.x;

        unsigned int N = A->numRows;

        if (i >= N || q >= N) return;

        float* fullyConstructedCRow = &fullyConstructedCMatrix[i * N];

	unsigned int kPtr = A->rowPtrs[i] + q;

	if (kPtr < A->rowPtrs[i + 1]) {
		unsigned int k = A->colIdxs[kPtr];
		
		for (unsigned int jPtr = B->rowPtrs[k]; jPtr < B->rowPtrs[k + 1]; ++jPtr) {
			unsigned int j = B->colIdxs[jPtr];

			// Accumulate dot products
			atomicAdd(&fullyConstructedCRow[j], A->values[kPtr] * B->values[jPtr]);
		}
	}
}

__global__ void spmspm_kernel1_write(
		const float* fullyConstructedCMatrix,
		COOMatrix* C) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int N = C->numRows;
	if (idx >= N * N) return;
	float dotProduct = fullyConstructedCMatrix[idx];
	if (dotProduct != 0) {
		unsigned int pos = atomicAdd(&C->numNonzeros, 1u);
                C->rowIdxs[pos] = idx / N;
                C->colIdxs[pos] = idx % N;
                C->values[pos] = dotProduct;
	}
}

void spmspm_gpu1(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
	int minGridSize;
	int blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, spmspm_kernel1);
	printf("minGridSize: %d, blokcSize: %d\n", minGridSize, blockSize);
	
	dim3 dimBlock(8, 8);
        unsigned int N = csrMatrix1->numRows;
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
        
	float* fullyConstructedCMatrix_d;
        cudaMalloc(&fullyConstructedCMatrix_d, sizeof(float) * N * N);
        cudaMemset(fullyConstructedCMatrix_d, 0, sizeof(float) * N * N);
	
	spmspm_kernel1<<<dimGrid, dimBlock>>>(csrMatrix1_d, csrMatrix2_d, fullyConstructedCMatrix_d);
	
	cudaDeviceSynchronize();

	unsigned int threadsPerBlock = 512;
	unsigned int numBlocks = (N * N + threadsPerBlock - 1) / threadsPerBlock;

	spmspm_kernel1_write<<<numBlocks, threadsPerBlock>>>(fullyConstructedCMatrix_d, cooMatrix_d);	
}

