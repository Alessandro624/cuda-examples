/* convolution1D.cu - 1D convolution examples using constant memory and tiling. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "../common/cli_utils.h"

#define DEBUG 0
#define FILTER_RADIUS 9
#define TILE_DIM 64
__constant__ float F_c[2 * FILTER_RADIUS + 1];

__global__ void convolution1DWithCacheAndTilingKernel(float *P, float *N, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float N_ds[TILE_DIM];
    if (i < n)
        N_ds[threadIdx.x] = N[i];
    else
        N_ds[threadIdx.x] = 0.0f;
    __syncthreads();
    if (i < n)
    {
        float Pvalue = 0.0f;
        for (int f = 0; f < (2 * FILTER_RADIUS + 1); f++)
        {
            int tile = threadIdx.x - FILTER_RADIUS + f;
            if (tile >= 0 && tile < TILE_DIM)
                Pvalue += F_c[f] * N_ds[tile];
            else
            {
                int normalIndex = i - FILTER_RADIUS + f;
                if (normalIndex >= 0 && normalIndex < n)
                    Pvalue += F_c[f] * N[normalIndex];
            }
        }
        P[i] = Pvalue;
    }
}

__global__ void convolution1DWithConstantMemoryKernel(float *P, float *N, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int start = i - FILTER_RADIUS;
        float Pvalue = 0.0f;
        for (int f = 0; f < 2 * FILTER_RADIUS + 1; f++)
        {
            if (start + f >= 0 && start + f < n)
                Pvalue += F_c[f] * N[start + f];
        }
        P[i] = Pvalue;
    }
}

void measureKernelExecution(void (*kernel)(float *, float *, int),
                            dim3 dimGrid, dim3 dimBlock, float *P_d, float *N_d,
                            int n, const char *kernelName)
{
    kernel<<<dimGrid, dimBlock>>>(P_d, N_d, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Kernel launch error for %s: %s\n", kernelName, cudaGetErrorString(err));
    cudaDeviceSynchronize();
    printf("Kernel %s executed\n", kernelName);
}

void convolution(float *P, float *N, float *F, int n)
{
    float *N_d, *P_d;
    int size = n * sizeof(float);
    int sizeF = (FILTER_RADIUS * 2 + 1) * sizeof(float);
    // allocating device memory
    cudaMalloc((void **)&N_d, size);
    cudaMalloc((void **)&P_d, size);
    // coping from host to device memory
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);
    // coping from host to constant memory
    cudaMemcpyToSymbol(F_c, F, sizeF);
    // number of blocks 1D (use helper to compute ceil(n / TILE_DIM))
    int blocks = compute_blocks_from_elements(n, TILE_DIM);
    dim3 dimGrid(blocks, 1, 1);
    // number of threads in each block
    dim3 dimBlock(64, 1, 1);
    // timing capture
    measureKernelExecution(convolution1DWithConstantMemoryKernel, dimGrid, dimBlock, P_d, N_d, n, "convolution1DWithConstantMemoryKernel");
    measureKernelExecution(convolution1DWithCacheAndTilingKernel, dimGrid, dimBlock, P_d, N_d, n, "convolution1DWithCacheAndTilingKernel");
    // coping from device to host memory
    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(N_d);
    cudaFree(P_d);
}

void printArray(float *A, int n)
{
    if (DEBUG)
    {
        printf("***********************\n");
        for (int i = 0; i < n; i++)
        {
            printf("%f ", A[i]);
        }
        printf("\n***********************\n");
    }
}

void fillArray(float *A, int n)
{
    for (int i = 0; i < n; i++)
    {
        float value = 1.0 * (rand() % 10);
        A[i] = value;
    }
}

/**
 * main - parse flags and run 1D convolution examples.
 */
int main(int argc, char **argv)
{
    if (cli_has_help(argc, argv))
    {
        printf("Usage: %s [--n N]\n", argv[0]);
        printf("  n: length of the 1D signal (default: 1000000)\n");
        return 0;
    }

    int n = 1000000;
    const char *v = cli_find_flag_value(argc, argv, "n");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid n '%s'\n", v);
            return 1;
        }
        n = atoi(v);
    }
    float *N_h, *F_h, *P_h;
    int size = n * sizeof(float);
    int sizeF = (FILTER_RADIUS * 2 + 1) * sizeof(float);
    // memory allocation for host
    N_h = (float *)malloc(size);
    F_h = (float *)malloc(sizeF);
    P_h = (float *)malloc(size);
    // arbitrary filling array (row major order)
    fillArray(N_h, n);
    fillArray(F_h, (FILTER_RADIUS * 2 + 1));
    printArray(N_h, n);
    printArray(F_h, (FILTER_RADIUS * 2 + 1));
    // calling convolution funcion
    convolution(P_h, N_h, F_h, n);
    // printing array
    printArray(P_h, n);
    // freeing host memory
    free(N_h);
    free(F_h);
    free(P_h);
    return 0;
}
