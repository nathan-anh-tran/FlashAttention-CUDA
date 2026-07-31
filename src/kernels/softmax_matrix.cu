#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax(const float* input, float* output, int cols) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // TODO: gi used to just be "which column" when there was one row.
    // Now blockIdx.x is which ROW this block owns, and tid is which column
    // within that row. Figure out the flattened index into a row-major
    // matrix: row * cols + col.
    int gi = blockIdx.x * cols + threadIdx.x;

    float val = input[gi];

    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float rowMax = sdata[0];

    float expVal = expf(val - rowMax);

    sdata[tid] = expVal;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float rowTotal = sdata[0];

    output[gi] = expVal / rowTotal;
}

int main() {
    const int rows = 3;
    const int cols = 8;
    const int n = rows * cols;


    float h_in[n] = {
        1.0f, 2.0f, 3.0f, 1000.0f, 5.0f, -2.0f, 1000.5f, 0.5f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    float h_out[n];

    float* d_in;
    float* d_out;

    cudaMalloc((void**) &d_in, sizeof(float) * n);
    cudaMalloc((void**) &d_out, sizeof(float) * n);
    
    cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);

    int numBlocks = rows;
    int threadsPerBlock = cols;
    int sharedMemBytes = threadsPerBlock * sizeof(float);

    softmax<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(d_in, d_out, cols);

    cudaMemcpy(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost);

    float h_exp[n];
    float h_expsums[rows];
    float rowMaxes[rows];
    for (int i = 0; i < rows; i += 1) {
        float rowMax = -INFINITY;
        for (int j = 0; j < cols; j += 1) {
            rowMax = max(rowMax, h_in[i * cols + j]);
        }
        rowMaxes[i] = rowMax;
    }
    for (int i = 0; i < rows; i += 1) {
        float h_expsum = 0.0f;
        for (int j = 0; j < cols; j += 1) {
            float elem = expf(h_in[i * cols + j] - rowMaxes[i]);
            h_exp[i * cols + j] = elem;
            h_expsum += elem;
        }
        h_expsums[i] = h_expsum;
    }

    bool passed = true;
    for (int i = 0; i < rows; i += 1) {
        for (int j = 0; j < cols; j += 1) {
            if (fabsf(h_exp[i * cols + j] / h_expsums[i] - h_out[i * cols + j]) > 1e-5) {
                passed = false;
            }
        }
    }
    if (passed) {
        std::cout << "Passed!" << '\n';
    } else {
        std::cout << "Failed" << '\n';
    }

    cudaFree(d_out);
    cudaFree(d_in);

    return 0;
}
