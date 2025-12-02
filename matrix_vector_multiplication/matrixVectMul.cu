/*
 * matrixVectMul.cu
 *
 * Simple matrix-vector multiplication example (C = A * B) using a per-row
 * parallelization: each thread computes one element of the output vector C.
 *
 * Usage:
 *  matrixVectMul [--width W] [--height H] [--threads T]
 *
 * - width:  number of columns of A (and length of vector B). Default: 1024
 * - height: number of rows of A (and length of output vector C). Default: 1024
 * - threads: threads per block for kernel launch. Default: 256
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../common/cli_utils.h"

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = (call);                                                                      \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)
#endif

#define DEBUG 0
#define WIDTH_DEFAULT (1 << 10)
#define HEIGHT_DEFAULT (1 << 10)
#define THREADS_DEFAULT 256

__global__ void matrixVectMulKernel(float *C, const float *A, const float *B, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < height)
    {
        float Cvalue = 0.0f;
        for (int k = 0; k < width; ++k)
            Cvalue += A[i * width + k] * B[k];
        C[i] = Cvalue;
    }
}

void matrixMul(float *C, float *A, float *B, int width, int height, int threads)
{
    float *A_d = NULL, *B_d = NULL, *C_d = NULL;
    size_t sizeA = (size_t)width * height * sizeof(float);
    size_t sizeB = (size_t)width * sizeof(float);
    size_t sizeC = (size_t)height * sizeof(float);

    // allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&A_d, sizeA));
    CHECK_CUDA(cudaMalloc((void **)&B_d, sizeB));
    CHECK_CUDA(cudaMalloc((void **)&C_d, sizeC));

    // copy from host to device
    CHECK_CUDA(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));

    // launch kernel
    threads = clamp_threads_to_device(threads);
    int blocks = compute_blocks_from_elements((long long)height, threads);

    matrixVectMulKernel<<<blocks, threads>>>(C_d, A_d, B_d, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy result back
    CHECK_CUDA(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    // free device memory
    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));
}

void printMatrix(const float *M, int width, int height)
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
            float value = 1.0f * (rand() % 10);
            M[i * width + j] = value;
        }
    }
}

int main(int argc, char **argv)
{
    if (cli_has_help(argc, argv))
    {
        printf("Usage: %s [--width W] [--height H] [--threads T]\n", argv[0]);
        printf("  All args optional. Defaults: width=%d height=%d threads=%d\n", WIDTH_DEFAULT, HEIGHT_DEFAULT, THREADS_DEFAULT);
        return 0;
    }

    int width_A = WIDTH_DEFAULT;
    int height_A = HEIGHT_DEFAULT;
    int threads = THREADS_DEFAULT;

    const char *v;
    v = cli_find_flag_value(argc, argv, "width");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid width '%s'\n", v);
            return 1;
        }
        width_A = atoi(v);
    }
    v = cli_find_flag_value(argc, argv, "height");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid height '%s'\n", v);
            return 1;
        }
        height_A = atoi(v);
    }
    v = cli_find_flag_value(argc, argv, "threads");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid threads '%s'\n", v);
            return 1;
        }
        threads = atoi(v);
    }

    if (threads <= 0)
        threads = THREADS_DEFAULT;

    float *A_h = NULL, *B_h = NULL, *C_h = NULL;
    size_t sizeA = (size_t)width_A * height_A * sizeof(float);
    size_t sizeB = (size_t)width_A * sizeof(float);
    size_t sizeC = (size_t)height_A * sizeof(float);

    // host allocations
    A_h = (float *)malloc(sizeA);
    B_h = (float *)malloc(sizeB);
    C_h = (float *)malloc(sizeC);

    if (!A_h || !B_h || !C_h)
    {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    // initialize
    fillMatrix(A_h, width_A, height_A);
    fillMatrix(B_h, width_A, 1);

    printMatrix(A_h, width_A, height_A);

    printf("Running matrix-vector multiplication: width=%d height=%d threads=%d\n", width_A, height_A, threads);

    matrixMul(C_h, A_h, B_h, width_A, height_A, threads);

    printMatrix(C_h, height_A, 1);

    free(A_h);
    free(B_h);
    free(C_h);
    return 0;
}
