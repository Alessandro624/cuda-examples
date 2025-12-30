/*
 * GPU Information Tool for Roofline Analysis
 *
 * Retrieves GPU specifications and calculates theoretical peak performance:
 * - Memory bandwidth (DRAM, L2, L1/Texture, Shared)
 * - Compute performance (FP32/FP64 GFLOP/s)
 * - Architecture details
 *
 * Outputs data compatible with roofline plotting tools
 *
 * Usage: ./gpu_info [device_id]
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Architecture-specific parameters
typedef struct
{
    int cores_per_sm;
    double fp64_ratio;
    double l2_bw_multiplier; // Relative to DRAM
    double shared_bw_gbps;   // Estimated shared memory bandwidth
    const char *arch_name;
} ArchSpec;

ArchSpec get_arch_spec(int major, int minor, const char *name)
{
    ArchSpec spec;

    // Detect architecture
    if (major == 3)
    {
        // Kepler
        spec.cores_per_sm = 192;
        spec.fp64_ratio = 1.0 / 3.0; // GK110
        spec.l2_bw_multiplier = 2.0;
        spec.shared_bw_gbps = 1500.0;
        spec.arch_name = "Kepler";
    }
    else if (major == 5)
    {
        // Maxwell
        spec.cores_per_sm = 128;
        spec.fp64_ratio = 1.0 / 32.0;
        spec.l2_bw_multiplier = 2.5;
        spec.shared_bw_gbps = 1800.0;
        spec.arch_name = "Maxwell";
    }
    else if (major == 6)
    {
        // Pascal
        spec.cores_per_sm = 64;
        if (strstr(name, "P100"))
        {
            spec.fp64_ratio = 1.0 / 2.0; // Tesla P100
        }
        else
        {
            spec.fp64_ratio = 1.0 / 32.0; // Consumer Pascal
        }
        spec.l2_bw_multiplier = 3.0;
        spec.shared_bw_gbps = 2000.0;
        spec.arch_name = "Pascal";
    }
    else if (major == 7)
    {
        // Volta/Turing
        spec.cores_per_sm = 64;
        if (strstr(name, "V100") || strstr(name, "Titan V"))
        {
            spec.fp64_ratio = 1.0 / 2.0; // Volta
        }
        else
        {
            spec.fp64_ratio = 1.0 / 32.0; // Turing
        }
        spec.l2_bw_multiplier = 3.5;
        spec.shared_bw_gbps = 2500.0;
        spec.arch_name = minor == 5 ? "Turing" : "Volta";
    }
    else if (major == 8)
    {
        // Ampere
        spec.cores_per_sm = 64;
        if (strstr(name, "A100") || strstr(name, "A40"))
        {
            spec.fp64_ratio = 1.0 / 2.0; // Data center
        }
        else
        {
            spec.fp64_ratio = 1.0 / 64.0; // Consumer (RTX 30xx)
        }
        spec.l2_bw_multiplier = 4.0;
        spec.shared_bw_gbps = 3000.0;
        spec.arch_name = "Ampere";
    }
    else if (major == 9)
    {
        // Hopper
        spec.cores_per_sm = 128;
        spec.fp64_ratio = 1.0 / 2.0;
        spec.l2_bw_multiplier = 5.0;
        spec.shared_bw_gbps = 4000.0;
        spec.arch_name = "Hopper";
    }
    else
    {
        // Unknown/future - use conservative defaults
        spec.cores_per_sm = 64;
        spec.fp64_ratio = 1.0 / 32.0;
        spec.l2_bw_multiplier = 3.0;
        spec.shared_bw_gbps = 2000.0;
        spec.arch_name = "Unknown";
    }

    return spec;
}

int main(int argc, char **argv)
{
    int device = 0;

    // Allow device selection
    if (argc > 1)
    {
        device = atoi(argv[1]);
    }

    // Get device count
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device >= device_count)
    {
        fprintf(stderr, "Error: Device %d not available (found %d devices)\n",
                device, device_count);
        exit(EXIT_FAILURE);
    }

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Get architecture specifications
    ArchSpec arch = get_arch_spec(prop.major, prop.minor, prop.name);

    // Calculate peak memory bandwidth (GB/s)
    // Formula: 2 * Memory_Clock (MHz) * (Bus_Width / 8) / 1000
    // Factor of 2 for DDR (Double Data Rate)
    double peak_bw_dram = 2.0 * (prop.memoryClockRate / 1000.0) *
                          (prop.memoryBusWidth / 8.0) / 1000.0;

    // Estimate cache/shared memory bandwidths
    double est_bw_l2 = peak_bw_dram * arch.l2_bw_multiplier;
    double est_bw_shared = arch.shared_bw_gbps;

    // Calculate number of CUDA cores
    int total_cores = prop.multiProcessorCount * arch.cores_per_sm;

    // Calculate peak GFLOP/s
    double clock_ghz = prop.clockRate / 1.0e6;

    // FP32: 2 FLOPs per clock per core (FMA = multiply + add)
    double peak_gflops_fp32 = 2.0 * total_cores * clock_ghz;
    double peak_gflops_fp64 = peak_gflops_fp32 * arch.fp64_ratio;

    // Print header
    printf("================================================================================\n");
    printf("GPU Device Information (Device %d)\n", device);
    printf("================================================================================\n\n");

    // Device information
    printf("Device Properties:\n");
    printf("------------------\n");
    printf("Name:                       %s\n", prop.name);
    printf("Architecture:               %s\n", arch.arch_name);
    printf("Compute Capability:         %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors (SMs):      %d\n", prop.multiProcessorCount);
    printf("CUDA Cores (estimated):     %d (%d per SM)\n",
           total_cores, arch.cores_per_sm);
    printf("Base Clock:                 %.0f MHz (%.3f GHz)\n",
           prop.clockRate / 1000.0, clock_ghz);
    printf("\n");

    // Memory hierarchy
    printf("Memory Hierarchy:\n");
    printf("-----------------\n");
    printf("Global Memory:              %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Memory Clock:               %.0f MHz\n",
           prop.memoryClockRate / 1000.0);
    printf("Memory Bus Width:           %d-bit\n", prop.memoryBusWidth);
    printf("L2 Cache Size:              %.2f MB\n",
           prop.l2CacheSize / (1024.0 * 1024.0));
    printf("Shared Memory per Block:    %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("Shared Memory per SM:       %zu KB\n",
           prop.sharedMemPerMultiprocessor / 1024);
    printf("Registers per Block:        %d\n", prop.regsPerBlock);
    printf("Registers per SM:           %d\n", prop.regsPerMultiprocessor);
    printf("Max Threads per Block:      %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM:         %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp Size:                  %d\n", prop.warpSize);
    printf("\n");

    // Bandwidth specifications (for roofline)
    printf("================================================================================\n");
    printf("Bandwidth Specifications (Theoretical Peak)\n");
    printf("================================================================================\n");
    printf("Global/DRAM:                %.1f GB/s\n", peak_bw_dram);
    printf("L2 Cache (est):             %.1f GB/s\n", est_bw_l2);
    printf("Shared Memory (est):        %.1f GB/s\n", est_bw_shared);
    printf("\n");

    // Compute specifications (for roofline)
    printf("================================================================================\n");
    printf("Compute Specifications (Theoretical Peak)\n");
    printf("================================================================================\n");
    printf("FP32 Performance:           %.1f GFLOP/s\n", peak_gflops_fp32);
    printf("FP64 Performance:           %.1f GFLOP/s (1/%.0f of FP32)\n",
           peak_gflops_fp64, 1.0 / arch.fp64_ratio);
    printf("\n");

    // Output for automated parsing (gnuplot compatible)
    printf("================================================================================\n");
    printf("Machine-Readable Output (for parse_metrics.py)\n");
    printf("================================================================================\n");
    printf("Global read: %.1f GB/s\n", peak_bw_dram);
    printf("Shared read: %.1f GB/s\n", est_bw_shared);
    printf("Texture read: %.1f GB/s\n", est_bw_l2);
    printf("Peak FP32: %.1f GFLOP/s\n", peak_gflops_fp32);
    printf("Peak FP64: %.1f GFLOP/s\n", peak_gflops_fp64);
    printf("\n");

    printf("Notes:\n");
    printf("------\n");
    printf("- These are THEORETICAL peak values from hardware specifications\n");
    printf("- Actual achievable bandwidth: typically 70-90%% of peak\n");
    printf("- L2 and Shared memory estimates based on architecture\n");
    printf("- Use microbenchmarks for empirical measurements\n");

    return 0;
}
