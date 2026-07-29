#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softmax(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gi = blockIdx.x * blockDim.x + threadIdx.x;

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
    const int n = 8;
    float h_in[n] = {1.0f, 2.0f, 3.0f, 1000.0f, 5.0f, -2.0f, 1000.5f, 0.5f};
    float h_out[n];

    float* d_in;
    float* d_out;

    cudaMalloc((void**) &d_in, sizeof(float) * n);
    cudaMalloc((void**) &d_out, sizeof(float) * n);

    cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);

    int numBlocks = 1;
    int threadsPerBlock = n;
    int sharedMemBytes = threadsPerBlock * sizeof(float);

    softmax<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost);

    float h_exp[n];
    float h_expsum = 0.0f;
    float rowMax = -INFINITY;
    for (int i = 0; i < n; i += 1) {
        rowMax = max(rowMax, h_in[i]);
    }
    for (int i = 0; i < n; i += 1) {
        float elem = expf(h_in[i] - rowMax);
        h_exp[i] = elem;
        h_expsum += elem;
    }

    bool passed = true;
    for (int i = 0; i < n; i += 1) {
        if (fabsf(h_exp[i] / h_expsum - h_out[i]) > 1e-5) {
            passed = false;
        }
    }
    if (passed) {
        std::cout << "Passed!" << '\n';
    } else {
        std::cout << "Failed" << '\n';
    }
    

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
