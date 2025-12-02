/*
 * vectAdd.cu
 * Clean, self-contained implementation with:
 *  - baseline kernel
 *  - grid-stride kernel with thread granularity
 *  - unified memory variants (with/without prefetch)
 *
 * Usage: ./vectAdd [--mode M] [--n N] [--threads T] [--granularity G]
 * Modes: 0=baseline, 1=grid-stride, 2=managed, 3=managed+prefetch
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../common/cli_utils.h"

#define DEFAULT_MODE 0
#define DEFAULT_N (1 << 20)
#define DEFAULT_THREADS 1024
#define DEFAULT_GRANULARITY 1

static inline void checkCuda(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
}
#define CHECK_CUDA(x) checkCuda((x), __FILE__, __LINE__)

__host__ __device__ static inline float addf(float a, float b) { return a + b; }

__global__ void vectAdd_baseline(float *C, const float *A, const float *B, int n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < (size_t)n)
        C[i] = addf(A[i], B[i]);
}

__global__ void vectAdd_grid_stride(float *C, const float *A, const float *B, int n, int granularity)
{
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    for (size_t base = tid; base < (size_t)n; base += stride * (size_t)granularity)
    {
        for (int g = 0; g < granularity; ++g)
        {
            size_t idx = base + (size_t)g * stride;
            if (idx < (size_t)n)
                C[idx] = addf(A[idx], B[idx]);
        }
    }
}

void fill_array(float *A, int n)
{
    for (int i = 0; i < n; ++i)
        A[i] = (float)(i & 0xFF);
}

void check_results(const float *C, const float *A, const float *B, int n)
{
    for (int i = 0; i < n; ++i)
    {
        float expect = A[i] + B[i];
        if (fabsf(C[i] - expect) > 1e-5f)
        {
            fprintf(stderr, "Mismatch at %d: got %f expected %f\n", i, C[i], expect);
            return;
        }
    }
    printf("Result check PASSED\n");
}

/**
 * main - parse flags, launch the requested vectAdd kernel variant and verify results.
 */
int main(int argc, char **argv)
{
    int mode = DEFAULT_MODE;
    int n = DEFAULT_N;
    int threads = DEFAULT_THREADS;
    int granularity = DEFAULT_GRANULARITY;

    if (cli_has_help(argc, argv))
    {
        printf("Usage: %s [--mode M] [--n N] [--threads T] [--granularity G]\n", argv[0]);
        printf("  Flags may be provided in any order. Defaults: mode=0 n=%d threads=%d granularity=%d\n", DEFAULT_N, DEFAULT_THREADS, DEFAULT_GRANULARITY);
        return 0;
    }

    const char *val;
    val = cli_find_flag_value(argc, argv, "mode");
    if (val)
    {
        if (!is_positive_integer_str(val))
        {
            fprintf(stderr, "Invalid value for --mode: %s\n", val);
            return 1;
        }
        mode = atoi(val);
    }
    val = cli_find_flag_value(argc, argv, "n");
    if (val)
    {
        if (!is_positive_integer_str(val))
        {
            fprintf(stderr, "Invalid value for --n: %s\n", val);
            return 1;
        }
        n = atoi(val);
    }
    val = cli_find_flag_value(argc, argv, "threads");
    if (val)
    {
        if (!is_positive_integer_str(val))
        {
            fprintf(stderr, "Invalid value for --threads: %s\n", val);
            return 1;
        }
        threads = atoi(val);
    }
    val = cli_find_flag_value(argc, argv, "granularity");
    if (val)
    {
        if (!is_positive_integer_str(val))
        {
            fprintf(stderr, "Invalid value for --granularity: %s\n", val);
            return 1;
        }
        granularity = atoi(val);
    }

    threads = clamp_threads_to_device(threads);
    int blocks = compute_blocks_from_elements((long long)n, threads);
    size_t bytes = (size_t)n * sizeof(float);
    printf("vectAdd mode=%d n=%d threads=%d blocks=%d granularity=%d\n", mode, n, threads, blocks, granularity);

    if (mode == 0 || mode == 1)
    {
        float *A = (float *)malloc(bytes);
        float *B = (float *)malloc(bytes);
        float *C = (float *)malloc(bytes);

        if (!A || !B || !C)
        {
            fprintf(stderr, "Host allocation failed for n=%d\n", n);
            return 1;
        }

        fill_array(A, n);
        fill_array(B, n);

        float *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc((void **)&dA, bytes));
        CHECK_CUDA(cudaMalloc((void **)&dB, bytes));
        CHECK_CUDA(cudaMalloc((void **)&dC, bytes));

        CHECK_CUDA(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));

        if (mode == 0)
            vectAdd_baseline<<<blocks, threads>>>(dC, dA, dB, n);
        else if (mode == 1)
            vectAdd_grid_stride<<<blocks, threads>>>(dC, dA, dB, n, granularity);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));

        check_results(C, A, B, n);

        free(A);
        free(B);
        free(C);

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));
    }
    else if (mode == 2 || mode == 3)
    {
        float *A, *B, *C;
        CHECK_CUDA(cudaMallocManaged((void **)&A, bytes));
        CHECK_CUDA(cudaMallocManaged((void **)&B, bytes));
        CHECK_CUDA(cudaMallocManaged((void **)&C, bytes));

        if (!A || !B || !C)
        {
            fprintf(stderr, "Managed allocation failed for n=%d\n", n);
            return 1;
        }

        for (int i = 0; i < n; ++i)
        {
            A[i] = (float)(i & 0xFF);
            B[i] = (float)(i & 0xFF);
            C[i] = 0.0f;
        }

        int dev = 0;
        CHECK_CUDA(cudaGetDevice(&dev));

        if (mode == 3)
        {
            CHECK_CUDA(cudaMemPrefetchAsync(A, bytes, dev, NULL));
            CHECK_CUDA(cudaMemPrefetchAsync(B, bytes, dev, NULL));
            CHECK_CUDA(cudaMemPrefetchAsync(C, bytes, dev, NULL));
        }

        vectAdd_grid_stride<<<blocks, threads>>>(C, A, B, n, granularity);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        if (mode == 3)
        {
            CHECK_CUDA(cudaMemPrefetchAsync(C, bytes, cudaCpuDeviceId, NULL));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        check_results(C, A, B, n);

        CHECK_CUDA(cudaFree(A));
        CHECK_CUDA(cudaFree(B));
        CHECK_CUDA(cudaFree(C));
    }
    else
    {
        fprintf(stderr, "Unknown mode %d\n", mode);
        return 1;
    }
    return 0;
}
