/*
 * errorCudaMemcpy.cu
 *
 * Small demo that intentionally contains a variety of cudaMemcpy / memory
 * management mistakes to illustrate runtime errors and sticky error states.
 * The purpose is pedagogical: do NOT use these patterns in production.
 *
 * Demonstrated issues (intentional):
 * - oversized host allocations (requests too much host memory)
 * - invalid cudaMemcpy source/destination pointers (nullptr or invalid host ptr)
 * - misuse of cudaMemcpyDeviceToDevice with host pointers
 * - not checking CUDA errors aggressively (this demo shows how errors appear)
 *
 * Build with the top-level Makefile target `errorCudaMemcpy` or simply `make`.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../common/cli_utils.h"
#include <cuda_runtime.h>

/**
 * CHECK_CUDA_ERROR - wrapper to report CUDA runtime errors.
 * This helper prints file/line information but does not exit; the demo
 * intentionally continues to show sticky error behaviour.
 */
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        printf("CUDA Runtime Error at: %s : %d\n", file, line);
        printf("%s %s\n", cudaGetErrorString(err), func);
        /* we intentionally do not exit here so the demo can show sticky errors */
    }
}

/**
 * CHECK_LAST_CUDA_ERROR - inspect last-launched-kernel error (peek) and report it.
 */
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, const int line)
{
    cudaError_t const err{cudaPeekAtLastError()};
    if (err != cudaSuccess)
    {
        printf("CUDA Runtime Error at: %s : %d\n", file, line);
        printf("%s\n", cudaGetErrorString(err));
    }
}

/**
 * RESET_LAST_CUDA_ERROR - consume the last error (cudaGetLastError) and report it.
 */
#define RESET_LAST_CUDA_ERROR() resetLast(__FILE__, __LINE__)
void resetLast(const char *const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        printf("CUDA Runtime Error at: %s : %d\n", file, line);
        printf("%s\n", cudaGetErrorString(err));
    }
}

#define nThread 256.0

/**
 * Simple vector-add kernel (correct behavior). The demo will intentionally
 * launch it with broken parameters in places to produce errors.
 */
__global__ void vectAddKernel(float *C, float *A, float *B, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

/**
 * vectAdd - allocate device memory, copy, launch kernel, and copy back.
 * This function intentionally performs several incorrect operations to
 * illustrate runtime errors (see comments inside).
 */
void vectAdd(float *C, float *A, float *B, int n)
{
    float *A_d, *B_d, *C_d, *D_d;
    int size = n * sizeof(float);
    /* allocating device memory (no checks here to show later failures) */
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);
    cudaMalloc((void **)&D_d, size);

    /* copying from host to device */
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    vectAddKernel<<<ceil(n / nThread), nThread>>>(C_d, A_d, B_d, n);
    cudaDeviceSynchronize();

    /*
     * Intentional wrong memcpy: copying from a null pointer into device
     * memory and using a device-to-device enum incorrectly. This should
     * produce a runtime error reported by CHECK_CUDA_ERROR.
     */
    CHECK_CUDA_ERROR(cudaMemcpy(D_d, nullptr, size, cudaMemcpyDeviceToDevice));

    CHECK_LAST_CUDA_ERROR();
    RESET_LAST_CUDA_ERROR();
    CHECK_LAST_CUDA_ERROR();

    /* another questionable launch to demonstrate sticky errors */
    vectAddKernel<<<ceil(n / nThread), nThread>>>(C_d, A_d, B_d, n);

    /* copy result back (may contain garbage if earlier failures occurred) */
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    /* free device memory */
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(D_d);
}

/**
 * main - parse flags and run the error demonstration for cudaMemcpy/memory mistakes.
 */
int main(int argc, char **argv)
{
    if (cli_has_help(argc, argv))
    {
        printf("Usage: %s [--n N]\n", argv[0]);
        printf("  n: number of elements to allocate/test (default: 1000000)\n");
        return 0;
    }

    /* intentionally large n to also illustrate oversized host allocation */
    int n = 1000000;
    const char *v = cli_find_flag_value(argc, argv, "n");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid n '%s' - must be positive integer\n", v);
            return 1;
        }
        n = atoi(v);
    }
    float *A_h, *B_h, *C_h;
    int size = n * sizeof(float);

    /* memory allocation for host arrays (may fail on small machines) */
    A_h = (float *)malloc(size);
    B_h = (float *)malloc(size);
    C_h = (float *)malloc(size);
    if (!A_h || !B_h || !C_h)
    {
        fprintf(stderr, "Host allocation failed for n=%d (size=%d).\n", n, size);
        return 1;
    }

    /* initialize arrays */
    for (int i = 0; i < n; i++)
    {
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }

    /* call the demo function that intentionally contains errors */
    vectAdd(C_h, A_h, B_h, n);

    bool success = true;
    /* naive verification; may fail if errors occurred earlier */
    for (int i = 0; i < n; i++)
    {
        if ((A_h[i] + B_h[i]) != C_h[i])
        {
            printf("Mismatch at %d: %f (expected %f)\n", i, C_h[i], A_h[i] + B_h[i]);
            success = false;
            break;
        }
    }
    if (success)
    {
        printf("Test PASSED!\n");
    }
    else
    {
        printf("Test FAILED!\n");
    }

    free(A_h);
    free(B_h);
    free(C_h);
    return 0;
}
