
#include "common.h"
#include "timer.h"

// Does not work, depends on ordered colIdxs.
/*
__global__ void spmspm_kernel0(CSRMatrix* A, CSRMatrix* B, COOMatrix* C) {
        unsigned int rowA = blockIdx.x * blockDim.x + threadIdx.x;
        if (rowA < A->numRows) {
                unsigned int beginRowA = A->rowPtrs[rowA];
                unsigned int endRowA = A->rowPtrs[rowA + 1];
                unsigned int bufferSize = endRowA - beginRowA;

                float valBufferA[bufferSize];
                unsigned int rowPtrBufferB[bufferSize];
                unsigned int rowPtrMaxesB[bufferSize];
                
                for (int i = beginRowA; i < endRowA; ++i) {
                        valBufferA[i - beginRowA] = A->values[i];
                        
                        rowPtrBufferB[i - beginRowA] = B->rowPtrs[A->colIdxs[i]];
                        rowPtrMaxesB[i - beginRowA] = B->rowPtrs[A->colIdxs[i] + 1];
                }

                while(true) {
                        float dotProduct = 0.0f;
                        unsigned int minCol = 0xFFFFFFFF;
                        for (int i = 0; i < bufferSize; ++i) {
                                if (rowPtrBufferB[i] < rowPtrMaxesB[i]) {
                                        unsigned int colIdxB = B->colIdxs[rowPtrBufferB[i]];
                                        if (colIdxB < minCol) {
                                                minCol = colIdxB;
                                                dotProduct = valBufferA[i] * B->values[rowPtrBufferB[i]];
                                        }
                                        else if (colIdxB == minCol) {
                                                dotProduct += valBufferA[i] * B->values[rowPtrBufferB[i]];
                                        }
                                }
                        }
                        for (int i = 0; i < bufferSize; ++i) {
                                if (rowPtrBufferB[i] < rowPtrMaxesB[i] && B->colIdxs[rowPtrBufferB[i]] == minCol) ++rowPtrBufferB[i];
                        }
                        if (dotProduct == 0.0f) break; // Assumes all matrix values are positive, same as is assumed in the CPU implementation
                        else {
                                unsigned int pos = atomicAdd(&C->numNonzeros, 1u);
                                C->rowIdxs[pos] = rowA;
                                C->colIdxs[pos] = minCol;
                                C->values[pos] = dotProduct;
                        }
                }
        }
}
*/

// Same as kernel below except one thread per row instead of output element.
// Much slower than kernel below.
__global__ void spmspm_kernel0_noarrays(CSRMatrix* A, CSRMatrix* B, COOMatrix* C) {
        // AB = C, where C_ij = sum_k{ A_ik B_kj } 
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= A->numRows) return;
        unsigned int kBeginPtr = A->rowPtrs[i];
        unsigned int kEndPtr = A->rowPtrs[i + 1];
        for (unsigned int j = 0; j < B->numCols; ++j) {
                float dotProduct = 0.0f;
                for (unsigned int kPtr = kBeginPtr; kPtr < kEndPtr; ++kPtr) {
                        unsigned int k = A->colIdxs[kPtr];
                        for (unsigned int jPtr = B->rowPtrs[k]; jPtr < B->rowPtrs[k + 1]; ++jPtr) {
                                unsigned int jNonzero = B->colIdxs[jPtr];
                                if (jNonzero == j) {
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
}

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


