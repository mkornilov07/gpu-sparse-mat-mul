
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
__global__ void spmspm_kernel3(
    const CSRMatrix* A,
    const CSRMatrix* B,
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

		unsigned int matchingThreads = __match_any_sync(__activemask(), j);
		float partialDotProduct = A->values[kPtr] * B->values[jPtr];
		if (lane == __reduce_min_sync(matchingThreads, lane)) {
			unsigned int counterMask = matchingThreads ^ (1u << (__builtin_ffs(matchingThreads) - 1));
			int srcLane = __builtin_ffs(counterMask) - 1;
			while (srcLane > 0) {
				partialDotProduct += __shfl_sync(matchingThreads, partialDotProduct, srcLane);
				counterMask ^= 1u << srcLane;
				srcLane = __builtin_ffs(counterMask) - 1;
			}
			fullyConstructedCRow[j] += partialDotProduct;
		}
		//for (int offset = 16; offset > 0; offset /= 2) partialDotProduct += __shfl_xor_sync(matchingThreads, partialDotProduct, offset);
		//partialDotProduct = __reduce_add_sync(matchingThreads, partialDotProduct);
		//if (lane == __reduce_min_sync(matchingThreads, lane)) fullyConstructedCRow[j] += partialDotProduct;

        	//atomicAdd(&fullyConstructedCRow[j], A->values[kPtr] * B->values[jPtr]);
    	}
    }
}

__global__ void spmspm_kernel3_write(
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



void spmspm_gpu3(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
	unsigned int N = csrMatrix1->numRows;
    unsigned int threadsPerBlock = 128;
    unsigned int warpsPerBlock = threadsPerBlock / 32;
    unsigned int numBlocks = (N + warpsPerBlock - 1) / warpsPerBlock;
        float* fullyConstructedCMatrix_d;
        cudaMalloc(&fullyConstructedCMatrix_d, sizeof(float) * N * N);
        cudaMemset(fullyConstructedCMatrix_d, 0, sizeof(float) * N * N);
        spmspm_kernel3<<<numBlocks, threadsPerBlock>>>(csrMatrix1_d, csrMatrix2_d, fullyConstructedCMatrix_d);
	cudaDeviceSynchronize();
	spmspm_kernel3_write<<<(N * N + 128 - 1) / 128, 128>>>(fullyConstructedCMatrix_d, cooMatrix_d);



















}

