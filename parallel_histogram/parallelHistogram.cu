/*
 * Parallel Histogram in CUDA
 *
 * Multiple kernel implementations demonstrating different optimization strategies:
 *   1. Naive (atomic on global memory)
 *   2. Privatization (per-block histogram in shared memory)
 *   3. Aggregation (consecutive same-value aggregation before atomic)
 *   4. Coarsened (each thread processes multiple elements)
 *
 * Usage:
 *   parallelHistogram [--mode MODE] [--n N] [--bins BINS] [--threads THREADS] [--coarse COARSE]
 *
 * - mode:    naive|privatized|aggregated|coarsened (default: privatized)
 * - n:       number of input elements (default: 1048576)
 * - bins:    number of histogram bins (default: 256)
 * - threads: threads per block (default: 256)
 * - coarse:  elements per thread for coarsened kernel (default: 4)
 *
 * Host-side verification is included.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#define N_DEFAULT (1 << 20) // 1M elements
#define BINS_DEFAULT 256
#define THREADS_DEFAULT 256
#define COARSE_FACTOR_DEFAULT 4
#define MAX_BINS 4096 // Maximum bins for shared memory privatization

/* ============================================================================
 * Kernel 1: Naive Histogram (Atomic on Global Memory)
 * Each thread processes one element and uses atomicAdd on global histogram.
 * Simple but suffers from atomic contention.
 * ============================================================================ */
__global__ void histogram_naive(const unsigned char *input, unsigned int *histogram, int n, int numBins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int bin = input[idx] % numBins;
        atomicAdd(&histogram[bin], 1);
    }
}

/* ============================================================================
 * Kernel 2: Privatized Histogram (Per-Block Shared Memory)
 * Each block maintains a private histogram in shared memory.
 * Reduces global memory atomic contention significantly.
 * Final step: merge block histograms to global.
 * ============================================================================ */
__global__ void histogram_privatized(const unsigned char *input, unsigned int *histogram, int n, int numBins)
{
    // Dynamically allocated shared memory for private histogram
    extern __shared__ unsigned int histo_s[];

    // Initialize shared memory histogram to zero
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        histo_s[i] = 0;
    }
    __syncthreads();

    // Each thread processes elements with grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int bin = input[i] % numBins;
        atomicAdd(&histo_s[bin], 1);
    }
    __syncthreads();

    // Merge shared histogram to global histogram
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        if (histo_s[i] > 0)
        {
            atomicAdd(&histogram[i], histo_s[i]);
        }
    }
}

/* ============================================================================
 * Kernel 3: Aggregated Histogram
 * Threads aggregate consecutive identical values before performing atomic.
 * Reduces number of atomic operations when input has local patterns.
 * ============================================================================ */
__global__ void histogram_aggregated(const unsigned char *input, unsigned int *histogram, int n, int numBins)
{
    extern __shared__ unsigned int histo_s[];

    // Initialize shared memory
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        histo_s[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Aggregation: accumulate count for same consecutive values
    int prevBin = -1;
    unsigned int count = 0;

    for (int i = idx; i < n; i += stride)
    {
        int bin = input[i] % numBins;

        if (bin == prevBin)
        {
            // Same bin as before, aggregate
            count++;
        }
        else
        {
            // Different bin, flush previous count
            if (count > 0)
            {
                atomicAdd(&histo_s[prevBin], count);
            }
            prevBin = bin;
            count = 1;
        }
    }

    // Flush remaining count
    if (count > 0)
    {
        atomicAdd(&histo_s[prevBin], count);
    }
    __syncthreads();

    // Merge to global
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        if (histo_s[i] > 0)
        {
            atomicAdd(&histogram[i], histo_s[i]);
        }
    }
}

/* ============================================================================
 * Kernel 4: Coarsened Histogram
 * Each thread processes multiple consecutive elements (coarsening factor).
 * Combines with privatization for best performance.
 * ============================================================================ */
__global__ void histogram_coarsened(const unsigned char *input, unsigned int *histogram,
                                    int n, int numBins, int coarseFactor)
{
    extern __shared__ unsigned int histo_s[];

    // Initialize shared memory
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        histo_s[i] = 0;
    }
    __syncthreads();

    // Each thread handles coarseFactor elements
    int baseIdx = (blockIdx.x * blockDim.x + threadIdx.x) * coarseFactor;

    // Local accumulator for consecutive same values
    int localBins[8]; // Support up to 8 coarse factor
    int localCounts[8];
    int localSize = 0;

    for (int c = 0; c < coarseFactor && (baseIdx + c) < n; c++)
    {
        int bin = input[baseIdx + c] % numBins;

        // Check if bin already in local accumulator
        int found = 0;
        for (int j = 0; j < localSize; j++)
        {
            if (localBins[j] == bin)
            {
                localCounts[j]++;
                found = 1;
                break;
            }
        }
        if (!found && localSize < 8)
        {
            localBins[localSize] = bin;
            localCounts[localSize] = 1;
            localSize++;
        }
        else if (!found)
        {
            // Overflow: flush to shared
            atomicAdd(&histo_s[bin], 1);
        }
    }

    // Flush local accumulator to shared
    for (int j = 0; j < localSize; j++)
    {
        atomicAdd(&histo_s[localBins[j]], localCounts[j]);
    }
    __syncthreads();

    // Merge to global
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        if (histo_s[i] > 0)
        {
            atomicAdd(&histogram[i], histo_s[i]);
        }
    }
}

/* ============================================================================
 * Host Reference Implementation
 * ============================================================================ */
void histogram_cpu(const unsigned char *input, unsigned int *histogram, int n, int numBins)
{
    memset(histogram, 0, numBins * sizeof(unsigned int));
    for (int i = 0; i < n; i++)
    {
        int bin = input[i] % numBins;
        histogram[bin]++;
    }
}

/* ============================================================================
 * Verification
 * ============================================================================ */
int verify_histogram(const unsigned int *gpu_hist, const unsigned int *cpu_hist, int numBins)
{
    for (int i = 0; i < numBins; i++)
    {
        if (gpu_hist[i] != cpu_hist[i])
        {
            fprintf(stderr, "Mismatch at bin %d: GPU=%u, CPU=%u\n", i, gpu_hist[i], cpu_hist[i]);
            return 0;
        }
    }
    return 1;
}

/* ============================================================================
 * Timing helper
 * ============================================================================ */
float run_kernel(void (*kernel)(const unsigned char *, unsigned int *, int, int),
                 const unsigned char *d_input, unsigned int *d_histogram,
                 int n, int numBins, int threadsPerBlock, int sharedMemSize,
                 const char *kernelName)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Clear histogram
    CHECK_CUDA(cudaMemset(d_histogram, 0, numBins * sizeof(unsigned int)));

    int blocks = compute_blocks_from_elements(n, threadsPerBlock);

    CHECK_CUDA(cudaEventRecord(start));
    kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_histogram, n, numBins);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel %s error: %s\n", kernelName, cudaGetErrorString(err));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

/* Overload for coarsened kernel */
float run_kernel_coarsened(const unsigned char *d_input, unsigned int *d_histogram,
                           int n, int numBins, int threadsPerBlock, int sharedMemSize,
                           int coarseFactor, const char *kernelName)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaMemset(d_histogram, 0, numBins * sizeof(unsigned int)));

    int elementsPerBlock = threadsPerBlock * coarseFactor;
    int blocks = (n + elementsPerBlock - 1) / elementsPerBlock;

    CHECK_CUDA(cudaEventRecord(start));
    histogram_coarsened<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_histogram, n, numBins, coarseFactor);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

/* ============================================================================
 * Print histogram (for debugging)
 * ============================================================================ */
void print_histogram(const unsigned int *histogram, int numBins, int maxPrint)
{
    printf("Histogram (first %d bins):\n", maxPrint);
    for (int i = 0; i < maxPrint && i < numBins; i++)
    {
        if (histogram[i] > 0)
            printf("  bin[%3d] = %u\n", i, histogram[i]);
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */
int main(int argc, char **argv)
{
    if (cli_has_help(argc, argv))
    {
        printf("Usage: %s [--mode MODE] [--n N] [--bins BINS] [--threads T] [--coarse C]\n\n", argv[0]);
        printf("Options:\n");
        printf("  --mode MODE    Kernel: naive|privatized|aggregated|coarsened|all (default: all)\n");
        printf("  --n N          Number of input elements (default: %d)\n", N_DEFAULT);
        printf("  --bins BINS    Number of histogram bins (default: %d)\n", BINS_DEFAULT);
        printf("  --threads T    Threads per block (default: %d)\n", THREADS_DEFAULT);
        printf("  --coarse C     Coarsening factor (default: %d)\n", COARSE_FACTOR_DEFAULT);
        return 0;
    }

    // Parse arguments
    int n = N_DEFAULT;
    int numBins = BINS_DEFAULT;
    int threadsPerBlock = THREADS_DEFAULT;
    int coarseFactor = COARSE_FACTOR_DEFAULT;
    const char *mode = "all";

    const char *v;
    if ((v = cli_find_flag_value(argc, argv, "n")))
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid n\n");
            return 1;
        }
        n = atoi(v);
    }
    if ((v = cli_find_flag_value(argc, argv, "bins")))
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid bins\n");
            return 1;
        }
        numBins = atoi(v);
    }
    if ((v = cli_find_flag_value(argc, argv, "threads")))
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid threads\n");
            return 1;
        }
        threadsPerBlock = atoi(v);
    }
    if ((v = cli_find_flag_value(argc, argv, "coarse")))
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid coarse\n");
            return 1;
        }
        coarseFactor = atoi(v);
    }
    if ((v = cli_find_flag_value(argc, argv, "mode")))
    {
        mode = v;
    }

    // Validate bins for shared memory
    if (numBins > MAX_BINS)
    {
        fprintf(stderr, "Warning: numBins=%d exceeds MAX_BINS=%d, using %d\n", numBins, MAX_BINS, MAX_BINS);
        numBins = MAX_BINS;
    }

    printf("=== Parallel Histogram ===\n");
    printf("Elements: %d\n", n);
    printf("Bins: %d\n", numBins);
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Coarsening factor: %d\n", coarseFactor);
    printf("Mode: %s\n\n", mode);

    // Allocate host memory
    size_t inputSize = n * sizeof(unsigned char);
    size_t histSize = numBins * sizeof(unsigned int);

    unsigned char *h_input = (unsigned char *)malloc(inputSize);
    unsigned int *h_histogram = (unsigned int *)malloc(histSize);
    unsigned int *h_histogram_cpu = (unsigned int *)malloc(histSize);

    // Initialize input with random values
    srand(42);
    for (int i = 0; i < n; i++)
    {
        h_input[i] = rand() % 256;
    }

    // Compute CPU reference
    histogram_cpu(h_input, h_histogram_cpu, n, numBins);

    // Allocate device memory
    unsigned char *d_input;
    unsigned int *d_histogram;
    CHECK_CUDA(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA(cudaMalloc(&d_histogram, histSize));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));

    int sharedMemSize = numBins * sizeof(unsigned int);
    float ms;
    int runAll = (strcmp(mode, "all") == 0);

    // Run kernels based on mode
    if (runAll || strcmp(mode, "naive") == 0)
    {
        ms = run_kernel(histogram_naive, d_input, d_histogram, n, numBins, threadsPerBlock, 0, "naive");
        CHECK_CUDA(cudaMemcpy(h_histogram, d_histogram, histSize, cudaMemcpyDeviceToHost));
        int ok = verify_histogram(h_histogram, h_histogram_cpu, numBins);
        printf("Kernel: histogram_naive\n");
        printf("  Time: %.3f ms\n", ms);
        printf("  Throughput: %.2f GElements/s\n", (n / 1e9) / (ms / 1e3));
        printf("  Verification: %s\n\n", ok ? "PASSED" : "FAILED");
    }

    if (runAll || strcmp(mode, "privatized") == 0)
    {
        ms = run_kernel(histogram_privatized, d_input, d_histogram, n, numBins, threadsPerBlock, sharedMemSize, "privatized");
        CHECK_CUDA(cudaMemcpy(h_histogram, d_histogram, histSize, cudaMemcpyDeviceToHost));
        int ok = verify_histogram(h_histogram, h_histogram_cpu, numBins);
        printf("Kernel: histogram_privatized\n");
        printf("  Time: %.3f ms\n", ms);
        printf("  Throughput: %.2f GElements/s\n", (n / 1e9) / (ms / 1e3));
        printf("  Verification: %s\n\n", ok ? "PASSED" : "FAILED");
    }

    if (runAll || strcmp(mode, "aggregated") == 0)
    {
        ms = run_kernel(histogram_aggregated, d_input, d_histogram, n, numBins, threadsPerBlock, sharedMemSize, "aggregated");
        CHECK_CUDA(cudaMemcpy(h_histogram, d_histogram, histSize, cudaMemcpyDeviceToHost));
        int ok = verify_histogram(h_histogram, h_histogram_cpu, numBins);
        printf("Kernel: histogram_aggregated\n");
        printf("  Time: %.3f ms\n", ms);
        printf("  Throughput: %.2f GElements/s\n", (n / 1e9) / (ms / 1e3));
        printf("  Verification: %s\n\n", ok ? "PASSED" : "FAILED");
    }

    if (runAll || strcmp(mode, "coarsened") == 0)
    {
        ms = run_kernel_coarsened(d_input, d_histogram, n, numBins, threadsPerBlock, sharedMemSize, coarseFactor, "coarsened");
        CHECK_CUDA(cudaMemcpy(h_histogram, d_histogram, histSize, cudaMemcpyDeviceToHost));
        int ok = verify_histogram(h_histogram, h_histogram_cpu, numBins);
        printf("Kernel: histogram_coarsened (factor=%d)\n", coarseFactor);
        printf("  Time: %.3f ms\n", ms);
        printf("  Throughput: %.2f GElements/s\n", (n / 1e9) / (ms / 1e3));
        printf("  Verification: %s\n\n", ok ? "PASSED" : "FAILED");
    }

    if (DEBUG)
    {
        print_histogram(h_histogram, numBins, 20);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_histogram);
    free(h_input);
    free(h_histogram);
    free(h_histogram_cpu);

    return 0;
}
