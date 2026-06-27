#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 100;
    // CPU allocations
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];

    // GPU allocations
    float* d_a;
    float* d_b;
    float* d_c;

    // Allocate space in GPU
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_c, n * sizeof(float));

    for (int i = 0; i < n; i += 1) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Copy over inputs from CPU to GPU
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel call
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy over output from GPU to CPU
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int i = 0; i < n; i += 1) {
        if (fabsf(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
            std::cout << "Failed at index " << i << '\n';
            passed = false;
            break;
        }
    }
    if (passed) {
        std::cout << "Passed!" << '\n';
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;


    return 0;
}