
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
__global__ void spmspm_kernel4(
                const CSRMatrix* A,
                const CSRMatrix* B,
                float* fullyConstructedCMatrix) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        unsigned int N = A->numRows;

	unsigned int i = idx / N;
	unsigned int q = idx % N;

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

__global__ void spmspm_kernel4_write(
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

void spmspm_gpu4(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
        int minGridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, spmspm_kernel4);

        unsigned int threadsPerBlock = blockSize;
        unsigned int N = csrMatrix1->numRows;
        unsigned int numBlocks = (N * N + threadsPerBlock - 1) / threadsPerBlock;

        float* fullyConstructedCMatrix_d;
        cudaMalloc(&fullyConstructedCMatrix_d, sizeof(float) * N * N);
        cudaMemset(fullyConstructedCMatrix_d, 0, sizeof(float) * N * N);

        spmspm_kernel4<<<numBlocks, threadsPerBlock>>>(csrMatrix1_d, csrMatrix2_d, fullyConstructedCMatrix_d);

        cudaDeviceSynchronize();

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, spmspm_kernel4_write);
        threadsPerBlock = blockSize;
        numBlocks = (N * N + threadsPerBlock - 1) / threadsPerBlock;

        spmspm_kernel4_write<<<numBlocks, threadsPerBlock>>>(fullyConstructedCMatrix_d, cooMatrix_d);
}
























