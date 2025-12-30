/*
 * deviceSpec.cu
 *
 * Collect and print CUDA device properties in a human-readable form.
 * This program enumerates all CUDA-capable devices and prints a
 * comprehensive set of properties useful for tuning kernels and
 * understanding hardware limits (threads per block, shared memory,
 * registers, clock rates, memory bandwidth, compute capability, etc.).
 *
 * Build: cd device_specification && make
 * Usage: ./deviceSpec [--device IDX]
 * If no index is provided, information for all devices is printed.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../common/cli_utils.h"
#include <cuda_runtime.h>

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

static void printDeviceProp(int idx, const cudaDeviceProp *p)
{
    printf("Device %d: %s\n", idx, p->name);
    printf("  Compute capability: %d.%d\n", p->major, p->minor);
    printf("  Total global memory: %.2f MB\n", p->totalGlobalMem / (1024.0f * 1024.0f));
    printf("  Multiprocessors (SM): %d\n", p->multiProcessorCount);
    printf("  CUDA cores (approx): %d\n", p->multiProcessorCount * (p->major >= 2 ? 192 : 32));
    printf("  Max threads per SM: %d\n", p->maxThreadsPerMultiProcessor);
    printf("  Max threads per block: %d\n", p->maxThreadsPerBlock);
    printf("  Max threads dim (block): x=%d y=%d z=%d\n", p->maxThreadsDim[0], p->maxThreadsDim[1], p->maxThreadsDim[2]);
    printf("  Max grid size: x=%d y=%d z=%d\n", p->maxGridSize[0], p->maxGridSize[1], p->maxGridSize[2]);
    printf("  Shared memory per block: %zu bytes\n", p->sharedMemPerBlock);
    printf("  Registers per block: %d\n", p->regsPerBlock);
    printf("  Warp size: %d\n", p->warpSize);
    printf("  Memory clock rate: %d kHz\n", p->memoryClockRate);
    printf("  Memory bus width: %d bits\n", p->memoryBusWidth);
    if (p->l2CacheSize)
        printf("  L2 cache size: %d bytes\n", p->l2CacheSize);
    printf("  Max texture1D: %d\n", p->maxTexture1D);
    printf("  Concurrent kernels: %s\n", p->concurrentKernels ? "yes" : "no");
    printf("  ECC enabled: %s\n", p->ECCEnabled ? "yes" : "no");
    printf("  PCI bus ID: %d  PCI device ID: %d\n", p->pciBusID, p->pciDeviceID);
    printf("\n");
}

/**
 * main - parse flags and print device properties for all or a selected device.
 */
int main(int argc, char **argv)
{
    int devCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&devCount));
    if (devCount == 0)
    {
        printf("No CUDA devices found.\n");
        return 0;
    }

    int queryIdx = -1;
    if (cli_has_help(argc, argv))
    {
        printf("Usage: %s [--device IDX]\n", argv[0]);
        printf("If no index is provided, information for all devices is printed.\n");
        return 0;
    }

    const char *v = cli_find_flag_value(argc, argv, "device");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid device index '%s'\n", v);
            return 1;
        }
        queryIdx = atoi(v);
    }

    for (int i = 0; i < devCount; ++i)
    {
        if (queryIdx >= 0 && queryIdx != i)
            continue;
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        printDeviceProp(i, &prop);
    }

    return 0;
}
