#include "common.h"
#include "timer.h"
// Mikhail
// Binary search through B

__global__ void bin_search_B_kernel(CSRMatrix* A, CSRMatrix* B, COOMatrix* C) {
    float sum = 0.; // sum A[i][k] * B[k][j]
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < A->numRows && j < B->numCols) {
        for(int csrPtr = A->rowPtrs[i]; csrPtr < A->rowPtrs[i+1]; csrPtr += 1) {
            int k = A->colIdxs[csrPtr];
            float valA = A->values[csrPtr];
            // need to find B[k, j] using binary search
            int bRowLeft = B->rowPtrs[k];
            int bRowRight = B->rowPtrs[k+1]; // [)
            // printf("bRowLeft = %d, bRowRight = %d, colIdxs[bRowLeft] = %d, colIdxs[bRowRight-1] = %d\n", bRowLeft, bRowRight, B->colIdxs[bRowLeft], B->colIdxs[bRowRight-1]);
            while(bRowRight > bRowLeft + 1) {
                int bRowMid = (bRowLeft + bRowRight) / 2;
                int midColIdx = B->colIdxs[bRowMid];
                if(midColIdx <= j) {
                    bRowLeft = bRowMid;
                }
                else {
                    bRowRight = bRowMid;
                }
            }
            if(B->colIdxs[bRowLeft] == j) {
                sum += B->values[bRowLeft] * valA;
            }
        }
        if(sum != 0.) { 
            //append (i, j, sum) to C atomically
            int insertIdx = atomicAdd(&(C->numNonzeros), 1);
            C->rowIdxs[insertIdx] = i;
            C->colIdxs[insertIdx] = j;
            C->values[insertIdx] = sum;
        }
    }
}


void spmspm_gpu1(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(csrMatrix1->numRows / threadsPerBlock.x + 1, csrMatrix2->numCols / threadsPerBlock.y + 1);
    bin_search_B_kernel <<< numBlocks, threadsPerBlock >>> (csrMatrix1_d, csrMatrix2_d, cooMatrix_d);


}




// struct CSRMatrix {
//     unsigned int numRows;
//     unsigned int numCols;
//     unsigned int numNonzeros;
//     unsigned int* rowPtrs;
//     unsigned int* colIdxs;
//     float* values;
// };



// struct COOMatrix {
//     unsigned int numRows;
//     unsigned int numCols;
//     unsigned int numNonzeros;
//     unsigned int capacity;
//     unsigned int* rowIdxs;
//     unsigned int* colIdxs;
//     float* values;
// };