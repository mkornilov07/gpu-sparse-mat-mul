#include "common.h"
#include "timer.h"

/*
 * Parallel implementation of SpMSpM with privatization.
 * One thread assigned to each row of A.
 * 
 * AB = C, where 
 * A, B, C are N-by-N,
 * A = B, and 
 * C_ij = sum_k{ A_ik * B_kj }
 */
__global__ void spmspm_kernel1(
                const CSRMatrix* A,
                const CSRMatrix* B,
                COOMatrix* C) {
	extern __shared__ float fullyConstructedCRows[];

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int N = A->numRows;
        if (i >= N) return;

	float* fullyConstructedCRow = &fullyConstructedCRows[threadIdx.x * N];
	for (unsigned int initPtr = 0; initPtr < N; ++initPtr) fullyConstructedCRow[initPtr] = 0.0f;

        for (unsigned int kPtr = A->rowPtrs[i]; kPtr < A->rowPtrs[i + 1]; ++kPtr) {
                unsigned int k = A->colIdxs[kPtr];

                for (unsigned int jPtr = B->rowPtrs[k]; jPtr < B->rowPtrs[k + 1]; ++jPtr) {
                        unsigned int j = B->colIdxs[jPtr];

                        // Accumulate dot products
                        fullyConstructedCRow[j] += A->values[kPtr] * B->values[jPtr];
                }
        }

        // Write nonzero dot products to COOMatrix C
        // Assumes all matrix elements in A and B are 
        // nonnegative (as is in the CPU implementation)
	
	//unsigned int numNonzeros = 0u;
	//for (unsigned int j = 0; j < N; ++j) if (fullyConstructedCRow[j] != 0.0f) ++numNonzeros;
	//unsigned int startingPos = atomicAdd(&C->numNonzeros, numNonzeros);
	//unsigned int nonzero = 0u;
        for (unsigned int j = 0; j < N; ++j) {
                float dotProduct = fullyConstructedCRow[j];
                if (dotProduct != 0.0f) {
                        unsigned int pos = atomicAdd(&C->numNonzeros, 1u);
                        C->rowIdxs[pos] = i;
                        C->colIdxs[pos] = j;
                        C->values[pos] = dotProduct;
                	//++nonzero;
		}
        }
}

void spmspm_gpu1(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	unsigned int N = csrMatrix1->numRows;
	unsigned int threadsPerBlock = prop.sharedMemPerBlock / (N * sizeof(float));
	if (threadsPerBlock > 0) {
		unsigned int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
		spmspm_kernel1<<<numBlocks, threadsPerBlock, threadsPerBlock * N * sizeof(float)>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d);
	}
	else {
		printf("(kernel1) Row(s) too large to fit in SM shared memory (N = %d, sharedMemPerBlock = %lu bytes. Defaulting to kernel0.\n", N, prop.sharedMemPerBlock);
		spmspm_gpu0(csrMatrix1, csrMatrix2, csrMatrix1_d, csrMatrix2_d, cooMatrix_d);
	}
}
