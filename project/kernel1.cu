#include "common.h"
#include "timer.h"
// Mikhail
// One thread per element of C, Linear search through row in B

#define DEBUG 0

__global__ void bin_search_B_kernel(CSRMatrix* A, CSRMatrix* B, COOMatrix* C) {
    float sum = 0.; // sum A[i][k] * B[k][j]
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < A->numRows && j < B->numCols) {
        for(int csrPtr = A->rowPtrs[i]; csrPtr < A->rowPtrs[i+1]; csrPtr += 1) {
            int k = A->colIdxs[csrPtr];
            float valA = A->values[csrPtr];
            if(DEBUG) printf("(%d, %d, %d): value in A is %f\n", i, j, k, valA);
            // need to find B[k, j] using binary search
            for(int bPtr = B->rowPtrs[k]; bPtr < B->rowPtrs[k+1]; bPtr++) {
                if(B->colIdxs[bPtr] == j) {
                if(DEBUG) printf("(%d, %d, %d): found matching column %d in B, the value there is %f, our A value is %f\n", i, j, k, j, B->values[bPtr], valA);
                sum += B->values[bPtr] * valA;
            }
            }
            
            }
            
        }
        if(sum != 0.) { 
            //append (i, j, sum) to C atomically
            if(DEBUG) printf("(%d, %d): writing sum %f\n", i, j, sum);
            int insertIdx = atomicAdd(&(C->numNonzeros), 1);
            C->rowIdxs[insertIdx] = i;
            C->colIdxs[insertIdx] = j;
            C->values[insertIdx] = sum;
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