/*
 * vectAdd_errors.cu
 * Demonstration of several common CUDA errors introduced one at a time.
 * Usage: ./vectAdd_errors <mode>
 * Modes:
 *  1 - excessive block size (invalid kernel launch configuration)
 *  2 - invalid host pointer passed to cudaMemcpy
 *  3 - excessive allocation request (cudaMalloc fails)
 *  4 - referencing invalid device pointer inside kernel (NULL device pointer)
 *  5 - out-of-bounds access in kernel (global memory overrun)
 *
 * Each mode intentionally triggers a different error so you can observe
 * runtime behaviour and the messages returned by CUDA runtime.
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include "../common/cli_utils.h"

#define CHECK_CUDA(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA ERROR: %s (at %s:%d)\n",        \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
        }                                                         \
    } while (0)

__global__ void vecAddKernel(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Simple addition; some modes intentionally pass invalid pointers or
        // write out-of-bounds to demonstrate errors.
        C[idx] = A[idx] + B[idx];
    }
}

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <mode>\n", prog);
    fprintf(stderr, "Modes:\n");
    fprintf(stderr, " 1 - excessive block size\n");
    fprintf(stderr, " 2 - invalid host pointer to cudaMemcpy\n");
    fprintf(stderr, " 3 - excessive allocation request\n");
    fprintf(stderr, " 4 - NULL device pointer referenced in kernel\n");
    fprintf(stderr, " 5 - kernel writes out-of-bounds (global mem access)\n");
}

/**
 * main - parse flags and run the selected error mode (0 = safe run).
 */
int main(int argc, char **argv)
{
    if (cli_has_help(argc, argv))
    {
        usage(argv[0]);
        return 0;
    }

    int mode = 0;
    /* Default problem size for safe runs and error demonstrations */
    long long N = 1024; // default number of elements for demos that need N

    const char *v = cli_find_flag_value(argc, argv, "mode");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid mode '%s' - must be a non-negative integer\n", v);
            return 1;
        }
        mode = atoi(v);
    }
    v = cli_find_flag_value(argc, argv, "n");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid n '%s' - must be a non-negative integer\n", v);
            return 1;
        }
        N = atoll(v);
    }

    printf("vectAdd_errors: running mode %d\n", mode);

    // Query device properties to craft errors that are reproducible
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // Allocate device buffers (may be NULL in some modes intentionally)
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    if (mode == 3)
    {
        // Mode 3: request an enormous allocation to force cudaMalloc failure
        size_t huge = (size_t)prop.totalGlobalMem * 1024ULL; // absurdly larger
        printf("Requesting huge allocation: %zu bytes (will likely fail)\n", huge);
        cudaError_t e = cudaMalloc((void **)&d_A, huge);
        if (e != cudaSuccess)
        {
            fprintf(stderr, "Expected cudaMalloc failure: %s\n", cudaGetErrorString(e));
            return 0;
        }
        // If it unexpectedly succeeds, free and return
        cudaFree(d_A);
        return 0;
    }

    // For other modes allocate normally
    size_t bytes = (size_t)N * sizeof(float);
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    // Fill device memory with something useful when needed
    if (mode != 2)
    {
        // Safe host buffers
        float *h_tmp = (float *)malloc(bytes);
        for (size_t i = 0; i < (size_t)N; ++i)
            h_tmp[i] = 1.0f;
        CHECK_CUDA(cudaMemcpy(d_A, h_tmp, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_tmp, bytes, cudaMemcpyHostToDevice));
        free(h_tmp);
    }

    if (mode == 1)
    {
        // Mode 1: excessive block size -> invalid configuration
        // Use an intentionally huge blockDim.x beyond device capability.
        int grid = 1;
        // craft a block size much larger than the device's limit
        int excessiveBlock = prop.maxThreadsPerBlock * 1024;
        printf("Launching kernel with blockDim.x=%d (maxThreadsPerBlock=%d)\n",
               excessiveBlock, prop.maxThreadsPerBlock);
        // This launch should produce a launch configuration error
        vecAddKernel<<<grid, excessiveBlock>>>(d_A, d_B, d_C, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch error (expected): %s\n", cudaGetErrorString(err));
        }
        else
        {
            printf("Unexpected: kernel launched without immediate error.\n");
        }
        // cleanup
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        return 0;
    }

    if (mode == 2)
    {
        // Mode 2: invalid pointer passed to cudaMemcpy
        // We'll attempt to copy from a deliberately invalid host pointer.
        float *bad_host_ptr = (float *)0x1; // invalid
        printf("Calling cudaMemcpy with invalid host pointer %p\n", (void *)bad_host_ptr);
        // Intentionally not using CHECK_CUDA so we can print the error returned
        cudaError_t err = cudaMemcpy(d_A, bad_host_ptr, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Expected cudaMemcpy error: %s\n", cudaGetErrorString(err));
        }
        else
        {
            fprintf(stderr, "Unexpected: cudaMemcpy succeeded with invalid pointer.\n");
        }
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        return 0;
    }

    if (mode == 4)
    {
        // Mode 4: reference an invalid device pointer in the kernel
        // Free one pointer to make it invalid (or set to NULL) then launch
        CHECK_CUDA(cudaFree(d_A));
        d_A = nullptr; // invalid device pointer intentionally
        printf("Launched kernel that will dereference a NULL device pointer.\n");
        vecAddKernel<<<128, 256>>>(d_A, d_B, d_C, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch/launch-time error: %s\n", cudaGetErrorString(err));
        }
        // Synchronize to catch device-side errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Device-side error (expected): %s\n", cudaGetErrorString(err));
        }
        else
        {
            printf("Unexpected: kernel finished without device-side error.\n");
        }
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        return 0;
    }

    if (mode == 5)
    {
        // Mode 5: allocate a small buffer but kernel writes out-of-bounds
        const size_t M = 16; // purposely tiny
        const size_t bytes_small = M * sizeof(float);
        printf("Allocating tiny device arrays (%zu elements) and launching a kernel that writes out-of-bounds.\n", M);
        // free previous allocations and allocate small ones
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaMalloc((void **)&d_A, bytes_small));
        CHECK_CUDA(cudaMalloc((void **)&d_B, bytes_small));
        CHECK_CUDA(cudaMalloc((void **)&d_C, bytes_small));

        // initialize small buffers on host
        float h_small[M];
        for (size_t i = 0; i < M; ++i)
            h_small[i] = 2.0f;
        CHECK_CUDA(cudaMemcpy(d_A, h_small, bytes_small, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_small, bytes_small, cudaMemcpyHostToDevice));

        // Launch kernel with N larger than M so kernel will write past d_C
        vecAddKernel<<<1, 256>>>(d_A, d_B, d_C, N); // N >> M
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        // Synchronize to catch device-side memory errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Device-side error (expected): %s\n", cudaGetErrorString(err));
        }
        else
        {
            printf("Unexpected: kernel completed without detecting out-of-bounds write.\n");
        }

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        return 0;
    }

    // Default: safe run (no intentional errors)
    printf("Running safe vector add (no intentional errors).\n");
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vecAddKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
