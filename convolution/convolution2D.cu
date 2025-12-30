/* convolution2D.cu - 2D convolution examples with tiling and constant memory variants. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "../common/cli_utils.h"

#define DEBUG 0
#define FILTER_RADIUS 9
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))
#define TILE_DIM 32
__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution2DWithCacheAndTilingKernel(float *P, float *N, int width, int height)
{
    int outCol = blockIdx.x * TILE_DIM + threadIdx.x;
    int outRow = blockIdx.y * TILE_DIM + threadIdx.y;
    __shared__ float N_ds[TILE_DIM][TILE_DIM];
    if (outRow < height && outCol < width)
        N_ds[threadIdx.y][threadIdx.x] = N[outRow * width + outCol];
    else
        N_ds[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    if (outRow < height && outCol < width)
    {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < (2 * FILTER_RADIUS + 1); fRow++)
        {
            for (int fCol = 0; fCol < (2 * FILTER_RADIUS + 1); fCol++)
            {
                int tileRow = threadIdx.y - FILTER_RADIUS + fRow;
                int tileCol = threadIdx.x - FILTER_RADIUS + fCol;
                if (tileRow >= 0 && tileRow < TILE_DIM && tileCol >= 0 && tileCol < TILE_DIM)
                    Pvalue += F_c[fRow][fCol] * N_ds[tileRow][tileCol];
                else
                {
                    int normalRow = outRow - FILTER_RADIUS + fRow;
                    int normalCol = outCol - FILTER_RADIUS + fCol;
                    if (normalRow >= 0 && normalRow < height && normalCol >= 0 && normalCol < width)
                        Pvalue += F_c[fRow][fCol] * N[normalRow * width + normalCol];
                }
            }
        }
        P[outRow * width + outCol] = Pvalue;
    }
}

__global__ void convolution2DWithTilingKernel(float *P, float *N, int width, int height)
{
    int outCol = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int outRow = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    __shared__ float N_ds[IN_TILE_DIM][IN_TILE_DIM];
    if (outRow >= 0 && outRow < height && outCol >= 0 && outCol < width)
        N_ds[threadIdx.y][threadIdx.x] = N[outRow * width + outCol];
    else
        N_ds[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    if (outRow >= 0 && outRow < height && outCol >= 0 && outCol < width)
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM)
        {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < (2 * FILTER_RADIUS + 1); fRow++)
            {
                for (int fCol = 0; fCol < (2 * FILTER_RADIUS + 1); fCol++)
                {
                    Pvalue += F_c[fRow][fCol] * N_ds[tileRow + fRow][tileCol + fCol];
                }
            }
            P[outRow * width + outCol] = Pvalue;
        }
}

__global__ void convolution2DWithConstantMemoryKernel(float *P, float *N, int width, int height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outRow < height && outCol < width)
    {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < (2 * FILTER_RADIUS + 1); fRow++)
        {
            for (int fCol = 0; fCol < (2 * FILTER_RADIUS + 1); fCol++)
            {
                int inRow = outRow - FILTER_RADIUS + fRow;
                int inCol = outCol - FILTER_RADIUS + fCol;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    Pvalue += F_c[fRow][fCol] * N[inRow * width + inCol];
            }
        }
        P[outRow * width + outCol] = Pvalue;
    }
}

__global__ void convolution2DBasicKernel(float *P, float *N, float *F, int r, int width, int height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outRow < height && outCol < width)
    {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < (2 * r + 1); fRow++)
        {
            for (int fCol = 0; fCol < (2 * r + 1); fCol++)
            {
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
            }
        }
        P[outRow * width + outCol] = Pvalue;
    }
}

void measureKernelExecution(void (*kernel1)(float *, float *, float *, int, int, int),
                            void (*kernel2)(float *, float *, int, int),
                            dim3 dimGrid, dim3 dimBlock, float *P_d, float *N_d, float *F_d,
                            int r, int width, int height, const char *kernelName)
{
    if (r > 0)
        kernel1<<<dimGrid, dimBlock>>>(P_d, N_d, F_d, r, width, height);
    else
        kernel2<<<dimGrid, dimBlock>>>(P_d, N_d, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Kernel launch error for %s: %s\n", kernelName, cudaGetErrorString(err));
    cudaDeviceSynchronize();
    printf("Kernel %s executed\n", kernelName);
}

void convolution(float *P, float *N, float *F, int r, int width, int height)
{
    float *N_d, *F_d, *P_d;
    int size = width * height * sizeof(float);
    int sizeF = (r * 2 + 1) * (r * 2 + 1) * sizeof(float);
    // allocating device memory
    cudaMalloc((void **)&N_d, size);
    cudaMalloc((void **)&F_d, sizeF);
    cudaMalloc((void **)&P_d, size);
    // coping from host to device memory
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F, sizeF, cudaMemcpyHostToDevice);
    // coping from host to constant memory
    cudaMemcpyToSymbol(F_c, F, sizeF);
    // number of blocks 2D (use helper to compute ceil(width/32), ceil(height/32))
    int bx = compute_blocks_from_elements(width, 32);
    int by = compute_blocks_from_elements(height, 32);
    dim3 dimGrid(bx, by, 1);
    // number of threads in each block
    dim3 dimBlock(32, 32, 1);
    // timing capture
    measureKernelExecution(convolution2DBasicKernel, nullptr, dimGrid, dimBlock, P_d, N_d, F_d, r, width, height, "convolution2DBasicKernel");
    measureKernelExecution(nullptr, convolution2DWithConstantMemoryKernel, dimGrid, dimBlock, P_d, N_d, nullptr, -1, width, height, "convolution2DWithConstantMemoryKernel");
    measureKernelExecution(nullptr, convolution2DWithTilingKernel, dimGrid, dimBlock, P_d, N_d, nullptr, -1, width, height, "convolution2DWithTilingKernel");
    measureKernelExecution(nullptr, convolution2DWithCacheAndTilingKernel, dimGrid, dimBlock, P_d, N_d, nullptr, -1, width, height, "convolution2DWithCacheAndTilingKernel");
    // coping from device to host memory
    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);
    // freeing device memory
    cudaFree(N_d);
    cudaFree(F_d);
    cudaFree(P_d);
}

void printMatrix(float *M, int width, int height)
{
    if (DEBUG)
    {
        printf("***********************\n");
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                printf("%f ", M[i * width + j]);
            }
            printf("\n");
        }
        printf("***********************\n");
    }
}

void fillMatrix(float *M, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float value = 1.0 * (rand() % 10);
            M[i * width + j] = value;
        }
    }
}

/**
 * main - parse flags and run 2D convolution examples.
 */
int main(int argc, char **argv)
{
    if (cli_has_help(argc, argv))
    {
        printf("Usage: %s [--width W] [--height H] [--radius R]\n", argv[0]);
        printf("  width,height: image dimensions (default: 1024 1024)\n");
        printf("  radius: filter radius (default: %d)\n", FILTER_RADIUS);
        return 0;
    }

    int width = 1024;
    int height = 1024;
    int r = FILTER_RADIUS;
    const char *v = cli_find_flag_value(argc, argv, "width");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid width '%s'\n", v);
            return 1;
        }
        width = atoi(v);
    }
    v = cli_find_flag_value(argc, argv, "height");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid height '%s'\n", v);
            return 1;
        }
        height = atoi(v);
    }
    v = cli_find_flag_value(argc, argv, "radius");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid radius '%s'\n", v);
            return 1;
        }
        r = atoi(v);
    }
    float *N_h, *F_h, *P_h;
    int size = width * height * sizeof(float);
    int sizeF = (r * 2 + 1) * (r * 2 + 1) * sizeof(float);
    // memory allocation for host
    N_h = (float *)malloc(size);
    F_h = (float *)malloc(sizeF);
    P_h = (float *)malloc(size);
    // arbitrary filling matrix (row major order)
    fillMatrix(N_h, width, height);
    fillMatrix(F_h, (r * 2 + 1), (r * 2 + 1));
    printMatrix(N_h, width, height);
    printMatrix(F_h, (r * 2 + 1), (r * 2 + 1));
    // calling convolution funcion
    convolution(P_h, N_h, F_h, r, width, height);
    // printing matrix
    printMatrix(P_h, width, height);
    // freeing host memory
    free(N_h);
    free(F_h);
    free(P_h);
    return 0;
}
