#include "common.h"
#include "timer.h"

// One thread per output element with linear search for colIdx.
__global__ void spmspm_kernel0_multidim(CSRMatrix* A, CSRMatrix* B, COOMatrix* C) {
        // AB = C, where C_ij = sum_k{ A_ik B_kj }
        unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= C->numRows || j >= C->numCols) return;
        float dotProduct = 0.0f;
        for (unsigned int kPtr = A->rowPtrs[i]; kPtr < A->rowPtrs[i + 1]; ++kPtr) {
                unsigned int k = A->colIdxs[kPtr];
                for (unsigned int jPtr = B->rowPtrs[k]; jPtr < B->rowPtrs[k + 1]; ++jPtr) {
                        unsigned int jVal = B->colIdxs[jPtr];
                        if (jVal == j) {
                                dotProduct += A->values[kPtr] * B->values[jPtr];
                                break;
                        }
                }
        }
        if (dotProduct > 0.0f) { // Assumes all matrix values are positive, same as is assumed in the CPU implementation
                unsigned int pos = atomicAdd(&C->numNonzeros, 1u);
                C->rowIdxs[pos] = i;
                C->colIdxs[pos] = j;
                C->values[pos] = dotProduct;
        }
}

void spmspm_gpu0(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
        printf("numrows: %d", csrMatrix1->numRows);
        // After testing, 16 x 16 seems to be optimal   
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((csrMatrix2->numCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (csrMatrix1->numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);
        spmspm_kernel0_multidim<<<numBlocks, threadsPerBlock>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d);
}
