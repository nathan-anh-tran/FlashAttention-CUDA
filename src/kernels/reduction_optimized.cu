#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void reduceSum(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    //Thread index
    int tid = threadIdx.x;
    //Global index
    int gi = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gi < n) ? input[gi] : 0.0f;
    __syncthreads();

    //Tree reduction to reduce all elements into index 0, large to small index
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

int main() {
    const int n = 1048576;
    float* h_in = new float[n];
    float* h_out = new float;

    float* d_in;
    float* d_out;

    cudaMalloc((void**) &d_in, sizeof(float) * n);
    cudaMalloc((void**) &d_out, sizeof(float));

    for (int i = 0; i < n; i += 1) {
        h_in[i] = i * 3 + 1;
    }

    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    int numBlocks = 4096;
    int threadsPerBlock = 256;
    int sharedMemBytes = sizeof(float) * threadsPerBlock;
    reduceSum<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    float h_sum = 0;
    for (int i = 0; i < n; i += 1) {
        h_sum += h_in[i];
    }

    if (fabsf(h_sum - *h_out) < 1e-5) {
        std::cout << "Passed!" << '\n';
    } else {
        std::cout << "Failed" << '\n';
    }

    cudaFree(d_in);
    cudaFree(d_out);

    delete[] h_in;
    delete h_out;

    return 0;
}