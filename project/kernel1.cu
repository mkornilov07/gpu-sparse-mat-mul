// Mikhail
// Binary search through B
#include "common.h"
#include "timer.h"



void spmspm_gpu1(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(csrMatrix1->numRows / threadsPerBlock.x + 1, csrMatrix2->numCols / threadsPerBlock.y + 1);
    bin_search_B_kernel <<< numBlocks, threadsPerBlock >>> (csrMatrix1_d, csrMatrix2_d, cooMatrix_d);


}

__global__ void bin_search_B_kernel(CSRMatrix* A, CSRMatrix* B, COOMatrix* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < A->numRows && j < B->numCols) {
        
    }
}


// struct CSRMatrix {
//     unsigned int numRows;
//     unsigned int numCols;
//     unsigned int numNonzeros;
//     unsigned int* rowPtrs;
//     unsigned int* colIdxs;
//     float* values;
// };