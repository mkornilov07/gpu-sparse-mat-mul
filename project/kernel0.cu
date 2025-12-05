
#include "common.h"
#include "timer.h"


#include "common.h"
#include "timer.h"
__global__
void spmspm_kernel0(const CSRMatrix* A, const CSRMatrix* B, 
                    COOMatrix* C, float* array) {

    // find which row to work in
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // check boundary condition
    if (row >= A->numRows) return;

    // array to accumulate vals for each col value in row
    float* valArray = &array[row * B->numCols];

    // initializing array
    for (int j = 0; j < B->numCols; j++)
        valArray[j] = 0.0f;

    // get start and end indices for cols
    int aStart = A->rowPtrs[row];
    int aEnd = A->rowPtrs[row + 1];

    for (int i = aStart; i < aEnd; i++) {
        int col_a = A->colIdxs[i];
        float aVal = A->values[i];

        // use col_a as rows of b
        int bStart = B->rowPtrs[col_a];
        int bEnd   = B->rowPtrs[col_a+1];

        for (int k = bStart; k < bEnd; k++) {
            int col_b = B->colIdxs[k];
            float bVal = B->values[k];
                    
            // accumulate dot product of A(i, j) * B(J, k) into valArray[k]
            valArray[col_b] += aVal * bVal;
        }
    }

     // write non zero valuees to COO matrix C
    for (int j = 0; j < B->numCols; j++) {
        float v = valArray[j];
        if (v != 0.0f) {
            int pos = atomicAdd(&C->numNonzeros, 1);
            C->rowIdxs[pos] = row;
            C->colIdxs[pos] = j;
            C->values[pos] = v;
        }
    }
}

void spmspm_gpu0(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
    
    int numThreads = 128;
    int numBlocks = (csrMatrix1->numRows + numThreads - 1) / numThreads;
    float* array_d;
    cudaMalloc(&array_d, sizeof(float) * csrMatrix1->numRows * csrMatrix2->numCols);
    cudaMemset(array_d, 0, sizeof(float) * csrMatrix1->numRows * csrMatrix2->numCols);

    spmspm_kernel0<<<numBlocks, numThreads>>>(csrMatrix1_d, csrMatrix2_d, cooMatrix_d, array_d);
    cudaDeviceSynchronize();

}


