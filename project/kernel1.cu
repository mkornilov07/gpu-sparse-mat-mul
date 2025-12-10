#include "common.h"
#include "timer.h"

/*
 * Parallel implementation of SpMSpM.
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
                COOMatrix* C,
                float* fullyConstructedCMatrix) {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        unsigned int N = A->numRows;

        if (i >= N) return;

        float* fullyConstructedCRow = &fullyConstructedCMatrix[i * N];

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
        for (unsigned int j = 0; j < N; ++j) {
                float dotProduct = fullyConstructedCRow[j];
                if (dotProduct != 0) {
                        unsigned int pos = atomicAdd(&C->numNonzeros, 1u);
                        C->rowIdxs[pos] = i;
                        C->colIdxs[pos] = j;
                        C->values[pos] = dotProduct;
                }
        }
}

void spmspm_gpu1(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
        unsigned int threadsPerBlock = 128;
        unsigned int N = csrMatrix1->numRows;
        unsigned int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        float* fullyConstructedCMatrix_d;
        cudaMalloc(&fullyConstructedCMatrix_d, sizeof(float) * N * N);
        cudaMemset(fullyConstructedCMatrix_d, 0, sizeof(float) * N * N);
        spmspm_kernel1<<<numBlocks, threadsPerBlock>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d, fullyConstructedCMatrix_d);
}

