#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#define TILE_WIDTH 16

__global__ void tiledMatmul(float* A, float* B, float* C, int m, int k, int n) {
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int i = row * n + col;
    // if (i < m * n) {
    //     float sum = 0.0;
    //     for (int j = 0; j < k; j += 1) {
    //         sum += A[row * k + j] * B[n * j + col];
    //     }
    //     C[i] = sum;
    // }
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    int numTiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t += 1) {
        int aCol = t * TILE_WIDTH + tx;
        int bRow = t * TILE_WIDTH + ty;

        if (row < m && aCol < k) {
            As[ty][tx] = A[row * k + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (bRow < k && col < n) {
            Bs[ty][tx] = B[bRow * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i += 1) {
            sum += (As[ty][i] * Bs[i][tx]);
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    //dimensions
    int m = 512;
    int k = 1024;
    int n = 512;

    //array backing for matrices in CPU --> C = AB
    float* h_A = new float[m * k];
    float* h_B = new float[k * n];
    float* h_C = new float[m * n];

    //matrices GPU
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc((void**) &d_A, sizeof(float) * m * k);
    cudaMalloc((void**) &d_B, sizeof(float) * k * n);
    cudaMalloc((void**) &d_C, sizeof(float) * m * n);

    for (int r = 0; r < m; r += 1) {
        for (int c = 0; c < k; c += 1) {
            int i = r * k + c;
            h_A[i] = 1.0f;
        }
    }
    for (int r = 0; r < k; r += 1) {
        for (int c = 0; c < n; c += 1) {
            int i = r * n + c;
            h_B[i] = 2.0f;
        }
    }

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + 15) / 16, (m + 15) / 16);
    tiledMatmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int r = 0; r < m; r += 1) {
        for (int c = 0; c < n; c += 1) {
            float sum = 0.0;
            for (int i = 0; i < k; i += 1) {
                sum += h_A[r * k + i] * h_B[n * i + c];
            }
            if (fabsf(h_C[r * n + c] - sum) > 1e-3) {
                passed = false;
            }
        }
    }
    if (passed) {
        std::cout << "Passed!" << '\n';
    } else {
        std::cout << "Failed!" << '\n';
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;


    return 0;
}