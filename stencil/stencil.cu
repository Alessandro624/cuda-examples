/*
 * 3D Seven-Point Stencil in CUDA
 *
 * Multiple kernel implementations demonstrating different optimization strategies:
 *   1. Naive (basic global memory access)
 *   2. Shared Memory Tiling (2D xy-plane tile cached in shared memory)
 *   3. Thread Coarsening (each thread processes multiple z-layers)
 *   4. Register Tiling (register caching along z-axis)
 *
 * The seven-point stencil computes:
 *   out[i,j,k] = c0*in[i,j,k]   + c1*in[i-1,j,k] + c2*in[i+1,j,k] +
 *                c3*in[i,j-1,k] + c4*in[i,j+1,k] +
 *                c5*in[i,j,k-1] + c6*in[i,j,k+1]
 *
 * Usage:
 *   stencil [--mode MODE] [--nx NX] [--ny NY] [--nz NZ] [--threads THREADS]
 *           [--tile-x TX] [--tile-y TY] [--coarse COARSE]
 *
 * - mode:    naive|shared|coarsened|register|all (default: all)
 * - nx/ny/nz: grid dimensions (default: 256x256x256)
 * - threads:  threads per block for naive kernel (default: 256)
 * - tile-x/tile-y: tile dimensions for tiled kernels (default: 32x8)
 * - coarse:  coarsening factor along z-axis (default: 8)
 *
 * Host-side verification is included.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
#define NX_DEFAULT 256
#define NY_DEFAULT 256
#define NZ_DEFAULT 256
#define THREADS_DEFAULT 256
#define TILE_X_DEFAULT 32
#define TILE_Y_DEFAULT 8
#define COARSE_FACTOR_DEFAULT 8

// Stencil coefficients (7-point stencil)
#define C0 -6.0f // center weight
#define C1 1.0f  // x-negative neighbor
#define C2 1.0f  // x-positive neighbor
#define C3 1.0f  // y-negative neighbor
#define C4 1.0f  // y-positive neighbor
#define C5 1.0f  // z-negative neighbor
#define C6 1.0f  // z-positive neighbor

/* ============================================================================
 * Kernel 1: Naive 3D Seven-Point Stencil
 * Direct global memory access. Each thread computes one output point.
 * Simple but memory bandwidth limited due to redundant loads.
 * ============================================================================ */
__global__ void stencil_naive(const float *in, float *out, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Skip boundary points
    if (i >= 1 && i < nx - 1 &&
        j >= 1 && j < ny - 1 &&
        k >= 1 && k < nz - 1)
    {
        // Linear index for 3D array stored in row-major (x fastest, then y, then z)
        // index = i + j*nx + k*nx*ny
        int idx = i + j * nx + k * nx * ny;
        int stride_x = 1;
        int stride_y = nx;
        int stride_z = nx * ny;

        float center = in[idx];
        float x_neg = in[idx - stride_x];
        float x_pos = in[idx + stride_x];
        float y_neg = in[idx - stride_y];
        float y_pos = in[idx + stride_y];
        float z_neg = in[idx - stride_z];
        float z_pos = in[idx + stride_z];

        out[idx] = C0 * center + C1 * x_neg + C2 * x_pos +
                   C3 * y_neg + C4 * y_pos + C5 * z_neg + C6 * z_pos;
    }
}

/* ============================================================================
 * Kernel 2: Shared Memory Tiling (xy-plane tiling)
 * 2D tiles in xy-plane are loaded into shared memory.
 * For each z-layer, load current plane + halo into shared memory.
 * Neighbors in z are loaded from global memory.
 * ============================================================================ */
template <int TILE_X, int TILE_Y>
__global__ void stencil_shared(const float *in, float *out, int nx, int ny, int nz)
{
    // Shared memory includes halo (+1 on each side in x and y)
    __shared__ float tile[TILE_Y + 2][TILE_X + 2];

    // Global coordinates
    int i = blockIdx.x * TILE_X + threadIdx.x;
    int j = blockIdx.y * TILE_Y + threadIdx.y;
    int k = blockIdx.z + 1; // Start from k=1 (skip boundary)

    // Local coordinates in shared memory (with halo offset)
    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;

    int stride_z = nx * ny;

    // Process only valid z-layers (interior)
    if (k >= 1 && k < nz - 1)
    {
        // Load center tile
        if (i < nx && j < ny)
        {
            int idx = i + j * nx + k * stride_z;
            tile[lj][li] = in[idx];
        }

        // Load x-halo (left and right)
        if (threadIdx.x == 0 && i > 0)
        {
            tile[lj][0] = in[(i - 1) + j * nx + k * stride_z];
        }
        if (threadIdx.x == TILE_X - 1 && i < nx - 1)
        {
            tile[lj][TILE_X + 1] = in[(i + 1) + j * nx + k * stride_z];
        }
        // Handle case where tile is at boundary
        if (threadIdx.x == blockDim.x - 1 && threadIdx.x < TILE_X - 1)
        {
            if (i + 1 < nx)
                tile[lj][li + 1] = in[(i + 1) + j * nx + k * stride_z];
        }

        // Load y-halo (top and bottom)
        if (threadIdx.y == 0 && j > 0)
        {
            tile[0][li] = in[i + (j - 1) * nx + k * stride_z];
        }
        if (threadIdx.y == TILE_Y - 1 && j < ny - 1)
        {
            tile[TILE_Y + 1][li] = in[i + (j + 1) * nx + k * stride_z];
        }
        if (threadIdx.y == blockDim.y - 1 && threadIdx.y < TILE_Y - 1)
        {
            if (j + 1 < ny)
                tile[lj + 1][li] = in[i + (j + 1) * nx + k * stride_z];
        }

        __syncthreads();

        // Compute stencil for interior points only
        if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
        {
            int idx = i + j * nx + k * stride_z;

            // xy-plane neighbors from shared memory
            float center = tile[lj][li];
            float x_neg = tile[lj][li - 1];
            float x_pos = tile[lj][li + 1];
            float y_neg = tile[lj - 1][li];
            float y_pos = tile[lj + 1][li];

            // z-neighbors from global memory
            float z_neg = in[idx - stride_z];
            float z_pos = in[idx + stride_z];

            out[idx] = C0 * center + C1 * x_neg + C2 * x_pos +
                       C3 * y_neg + C4 * y_pos + C5 * z_neg + C6 * z_pos;
        }
    }
}

/* ============================================================================
 * Kernel 3: Thread Coarsening (z-axis coarsening)
 * Each thread processes multiple consecutive z-layers.
 * Reduces thread launch overhead and improves data reuse along z.
 * ============================================================================ */
template <int TILE_X, int TILE_Y, int COARSE>
__global__ void stencil_coarsened(const float *in, float *out, int nx, int ny, int nz)
{
    __shared__ float tile[TILE_Y + 2][TILE_X + 2];

    int i = blockIdx.x * TILE_X + threadIdx.x;
    int j = blockIdx.y * TILE_Y + threadIdx.y;
    int k_base = blockIdx.z * COARSE + 1; // Start from k=1

    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;

    int stride_z = nx * ny;

    // Each thread processes COARSE z-layers
    for (int c = 0; c < COARSE; c++)
    {
        int k = k_base + c;

        if (k >= nz - 1)
            break; // Beyond valid range

        // Load tile for current z-layer
        if (i < nx && j < ny)
        {
            int idx = i + j * nx + k * stride_z;
            tile[lj][li] = in[idx];
        }

        // Load x-halo
        if (threadIdx.x == 0 && i > 0)
        {
            tile[lj][0] = in[(i - 1) + j * nx + k * stride_z];
        }
        if (threadIdx.x == TILE_X - 1 && i < nx - 1)
        {
            tile[lj][TILE_X + 1] = in[(i + 1) + j * nx + k * stride_z];
        }

        // Load y-halo
        if (threadIdx.y == 0 && j > 0)
        {
            tile[0][li] = in[i + (j - 1) * nx + k * stride_z];
        }
        if (threadIdx.y == TILE_Y - 1 && j < ny - 1)
        {
            tile[TILE_Y + 1][li] = in[i + (j + 1) * nx + k * stride_z];
        }

        __syncthreads();

        // Compute stencil
        if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
        {
            int idx = i + j * nx + k * stride_z;

            float center = tile[lj][li];
            float x_neg = tile[lj][li - 1];
            float x_pos = tile[lj][li + 1];
            float y_neg = tile[lj - 1][li];
            float y_pos = tile[lj + 1][li];
            float z_neg = in[idx - stride_z];
            float z_pos = in[idx + stride_z];

            out[idx] = C0 * center + C1 * x_neg + C2 * x_pos +
                       C3 * y_neg + C4 * y_pos + C5 * z_neg + C6 * z_pos;
        }

        __syncthreads();
    }
}

/* ============================================================================
 * Kernel 4: Register Tiling (z-axis register caching)
 * As we sweep through z-layers, cache values in registers.
 * Each thread maintains registers for prev, curr, next z-values.
 * Maximizes temporal reuse along z-dimension.
 * ============================================================================ */
template <int TILE_X, int TILE_Y>
__global__ void stencil_register(const float *in, float *out, int nx, int ny, int nz)
{
    __shared__ float tile[TILE_Y + 2][TILE_X + 2];

    int i = blockIdx.x * TILE_X + threadIdx.x;
    int j = blockIdx.y * TILE_Y + threadIdx.y;

    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;

    int stride_z = nx * ny;

    if (i >= nx || j >= ny)
        return;

    // Register variables for z-values (sliding window)
    float z_prev, z_curr, z_next;

    // Initialize: load first two z-layers into registers
    z_prev = in[i + j * nx + 0 * stride_z];
    z_curr = in[i + j * nx + 1 * stride_z];

    // Sweep through z-layers from k=1 to k=nz-2
    for (int k = 1; k < nz - 1; k++)
    {
        // Prefetch next z-layer
        z_next = in[i + j * nx + (k + 1) * stride_z];

        // Load xy-tile for current z into shared memory
        tile[lj][li] = z_curr;

        // Load x-halo
        if (threadIdx.x == 0 && i > 0)
        {
            tile[lj][0] = in[(i - 1) + j * nx + k * stride_z];
        }
        if (threadIdx.x == TILE_X - 1 || threadIdx.x == blockDim.x - 1)
        {
            if (i + 1 < nx)
                tile[lj][li + 1] = in[(i + 1) + j * nx + k * stride_z];
        }

        // Load y-halo
        if (threadIdx.y == 0 && j > 0)
        {
            tile[0][li] = in[i + (j - 1) * nx + k * stride_z];
        }
        if (threadIdx.y == TILE_Y - 1 || threadIdx.y == blockDim.y - 1)
        {
            if (j + 1 < ny)
                tile[lj + 1][li] = in[i + (j + 1) * nx + k * stride_z];
        }

        __syncthreads();

        // Compute stencil for interior points
        if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
        {
            int idx = i + j * nx + k * stride_z;

            // xy-neighbors from shared memory
            float x_neg = tile[lj][li - 1];
            float x_pos = tile[lj][li + 1];
            float y_neg = tile[lj - 1][li];
            float y_pos = tile[lj + 1][li];

            // z-neighbors from registers!
            out[idx] = C0 * z_curr + C1 * x_neg + C2 * x_pos +
                       C3 * y_neg + C4 * y_pos + C5 * z_prev + C6 * z_next;
        }

        __syncthreads();

        // Slide register window
        z_prev = z_curr;
        z_curr = z_next;
    }
}

/* ============================================================================
 * Host Reference Implementation
 * ============================================================================ */
void stencil_cpu(const float *in, float *out, int nx, int ny, int nz)
{
    for (int k = 1; k < nz - 1; k++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = 1; i < nx - 1; i++)
            {
                int idx = i + j * nx + k * nx * ny;
                int stride_x = 1;
                int stride_y = nx;
                int stride_z = nx * ny;

                float center = in[idx];
                float x_neg = in[idx - stride_x];
                float x_pos = in[idx + stride_x];
                float y_neg = in[idx - stride_y];
                float y_pos = in[idx + stride_y];
                float z_neg = in[idx - stride_z];
                float z_pos = in[idx + stride_z];

                out[idx] = C0 * center + C1 * x_neg + C2 * x_pos +
                           C3 * y_neg + C4 * y_pos + C5 * z_neg + C6 * z_pos;
            }
        }
    }
}

/* ============================================================================
 * Verification
 * ============================================================================ */
int verify_stencil(const float *gpu_out, const float *cpu_out, int nx, int ny, int nz, float tolerance)
{
    int errors = 0;
    for (int k = 1; k < nz - 1; k++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            for (int i = 1; i < nx - 1; i++)
            {
                int idx = i + j * nx + k * nx * ny;
                float diff = fabsf(gpu_out[idx] - cpu_out[idx]);
                if (diff > tolerance)
                {
                    if (errors < 5)
                    {
                        fprintf(stderr, "Mismatch at [%d,%d,%d]: GPU=%.6f, CPU=%.6f, diff=%.6e\n",
                                i, j, k, gpu_out[idx], cpu_out[idx], diff);
                    }
                    errors++;
                }
            }
        }
    }
    if (errors > 0)
    {
        fprintf(stderr, "Total mismatches: %d\n", errors);
    }
    return (errors == 0);
}

/* ============================================================================
 * Timing helper
 * ============================================================================ */
typedef void (*kernel_launcher_t)(const float *, float *, int, int, int, cudaEvent_t, cudaEvent_t);

float run_and_time(const float *d_in, float *d_out, int nx, int ny, int nz,
                   kernel_launcher_t launcher, const char *name)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Clear output
    CHECK_CUDA(cudaMemset(d_out, 0, (size_t)nx * ny * nz * sizeof(float)));

    launcher(d_in, d_out, nx, ny, nz, start, stop);

    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel %s error: %s\n", name, cudaGetErrorString(err));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

/* ============================================================================
 * Kernel launchers
 * ============================================================================ */
void launch_naive(const float *d_in, float *d_out, int nx, int ny, int nz,
                  cudaEvent_t start, cudaEvent_t stop)
{
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (nz + blockDim.z - 1) / blockDim.z);

    CHECK_CUDA(cudaEventRecord(start));
    stencil_naive<<<gridDim, blockDim>>>(d_in, d_out, nx, ny, nz);
    CHECK_CUDA(cudaEventRecord(stop));
}

#define TILE_X 32
#define TILE_Y 8

void launch_shared(const float *d_in, float *d_out, int nx, int ny, int nz,
                   cudaEvent_t start, cudaEvent_t stop)
{
    dim3 blockDim(TILE_X, TILE_Y, 1);
    dim3 gridDim((nx + TILE_X - 1) / TILE_X,
                 (ny + TILE_Y - 1) / TILE_Y,
                 nz - 2); // One block per interior z-layer

    CHECK_CUDA(cudaEventRecord(start));
    stencil_shared<TILE_X, TILE_Y><<<gridDim, blockDim>>>(d_in, d_out, nx, ny, nz);
    CHECK_CUDA(cudaEventRecord(stop));
}

#define COARSE_Z 8

void launch_coarsened(const float *d_in, float *d_out, int nx, int ny, int nz,
                      cudaEvent_t start, cudaEvent_t stop)
{
    dim3 blockDim(TILE_X, TILE_Y, 1);
    int z_blocks = (nz - 2 + COARSE_Z - 1) / COARSE_Z;
    dim3 gridDim((nx + TILE_X - 1) / TILE_X,
                 (ny + TILE_Y - 1) / TILE_Y,
                 z_blocks);

    CHECK_CUDA(cudaEventRecord(start));
    stencil_coarsened<TILE_X, TILE_Y, COARSE_Z><<<gridDim, blockDim>>>(d_in, d_out, nx, ny, nz);
    CHECK_CUDA(cudaEventRecord(stop));
}

void launch_register(const float *d_in, float *d_out, int nx, int ny, int nz,
                     cudaEvent_t start, cudaEvent_t stop)
{
    dim3 blockDim(TILE_X, TILE_Y, 1);
    dim3 gridDim((nx + TILE_X - 1) / TILE_X,
                 (ny + TILE_Y - 1) / TILE_Y,
                 1); // Single z-block, thread sweeps all z

    CHECK_CUDA(cudaEventRecord(start));
    stencil_register<TILE_X, TILE_Y><<<gridDim, blockDim>>>(d_in, d_out, nx, ny, nz);
    CHECK_CUDA(cudaEventRecord(stop));
}

/* ============================================================================
 * Main
 * ============================================================================ */
int main(int argc, char **argv)
{
    if (cli_has_help(argc, argv))
    {
        printf("Usage: %s [--mode MODE] [--nx NX] [--ny NY] [--nz NZ]\n\n", argv[0]);
        printf("Options:\n");
        printf("  --mode MODE    Kernel: naive|shared|coarsened|register|all (default: all)\n");
        printf("  --nx NX        Grid size in X (default: %d)\n", NX_DEFAULT);
        printf("  --ny NY        Grid size in Y (default: %d)\n", NY_DEFAULT);
        printf("  --nz NZ        Grid size in Z (default: %d)\n", NZ_DEFAULT);
        printf("\n");
        printf("Stencil coefficients:\n");
        printf("  c0 = %.2f (center), c1 = %.2f (each neighbor)\n", C0, C1);
        printf("\n");
        printf("Tile configuration:\n");
        printf("  Shared/Coarsened/Register: %dx%d xy-tile\n", TILE_X_DEFAULT, TILE_Y_DEFAULT);
        printf("  Coarsening factor: %d z-layers per thread\n", COARSE_FACTOR_DEFAULT);
        return 0;
    }

    // Parse arguments
    int nx = NX_DEFAULT;
    int ny = NY_DEFAULT;
    int nz = NZ_DEFAULT;
    const char *mode = "all";

    const char *v;
    if ((v = cli_find_flag_value(argc, argv, "nx")))
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid nx\n");
            return 1;
        }
        nx = atoi(v);
    }
    if ((v = cli_find_flag_value(argc, argv, "ny")))
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid ny\n");
            return 1;
        }
        ny = atoi(v);
    }
    if ((v = cli_find_flag_value(argc, argv, "nz")))
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid nz\n");
            return 1;
        }
        nz = atoi(v);
    }
    if ((v = cli_find_flag_value(argc, argv, "mode")))
    {
        mode = v;
    }

    // Validate minimum size
    if (nx < 3 || ny < 3 || nz < 3)
    {
        fprintf(stderr, "Grid must be at least 3x3x3 for stencil computation\n");
        return 1;
    }

    size_t totalElements = (size_t)nx * ny * nz;
    size_t dataSize = totalElements * sizeof(float);
    size_t interiorPoints = (size_t)(nx - 2) * (ny - 2) * (nz - 2);

    printf("=== 3D Seven-Point Stencil ===\n");
    printf("Grid: %d x %d x %d = %zu elements\n", nx, ny, nz, totalElements);
    printf("Interior points: %zu\n", interiorPoints);
    printf("Data size: %.2f MB\n", dataSize / (1024.0 * 1024.0));
    printf("Mode: %s\n\n", mode);

    // Allocate host memory
    float *h_in = (float *)malloc(dataSize);
    float *h_out_gpu = (float *)malloc(dataSize);
    float *h_out_cpu = (float *)malloc(dataSize);

    if (!h_in || !h_out_gpu || !h_out_cpu)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    // Initialize input with random values
    srand(42);
    for (size_t i = 0; i < totalElements; i++)
    {
        h_in[i] = (float)(rand() % 100) / 100.0f;
    }
    memset(h_out_cpu, 0, dataSize);

    // Compute CPU reference
    printf("Computing CPU reference...\n");
    stencil_cpu(h_in, h_out_cpu, nx, ny, nz);

    // Allocate device memory
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, dataSize));
    CHECK_CUDA(cudaMalloc(&d_out, dataSize));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, dataSize, cudaMemcpyHostToDevice));

    float ms;
    int runAll = (strcmp(mode, "all") == 0);
    float tolerance = 1e-5f;

    // Run kernels based on mode
    if (runAll || strcmp(mode, "naive") == 0)
    {
        ms = run_and_time(d_in, d_out, nx, ny, nz, launch_naive, "stencil_naive");
        CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, dataSize, cudaMemcpyDeviceToHost));
        int ok = verify_stencil(h_out_gpu, h_out_cpu, nx, ny, nz, tolerance);

        double flops = interiorPoints * 13.0; // 7 loads, 6 adds, 7 multiplies -> ~13 FLOPs
        double gflops = (flops / 1e9) / (ms / 1e3);
        double bandwidth = (interiorPoints * 7 * sizeof(float) + interiorPoints * sizeof(float)) / 1e9 / (ms / 1e3);

        printf("Kernel: stencil_naive\n");
        printf("  Time: %.3f ms\n", ms);
        printf("  Throughput: %.2f GPoints/s\n", (interiorPoints / 1e9) / (ms / 1e3));
        printf("  Est. GFLOP/s: %.2f\n", gflops);
        printf("  Est. Bandwidth: %.2f GB/s\n", bandwidth);
        printf("  Verification: %s\n\n", ok ? "PASSED" : "FAILED");
    }

    if (runAll || strcmp(mode, "shared") == 0)
    {
        ms = run_and_time(d_in, d_out, nx, ny, nz, launch_shared, "stencil_shared");
        CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, dataSize, cudaMemcpyDeviceToHost));
        int ok = verify_stencil(h_out_gpu, h_out_cpu, nx, ny, nz, tolerance);

        double gflops = (interiorPoints * 13.0 / 1e9) / (ms / 1e3);
        double bandwidth = (interiorPoints * 7 * sizeof(float) + interiorPoints * sizeof(float)) / 1e9 / (ms / 1e3);

        printf("Kernel: stencil_shared (tile %dx%d)\n", TILE_X, TILE_Y);
        printf("  Time: %.3f ms\n", ms);
        printf("  Throughput: %.2f GPoints/s\n", (interiorPoints / 1e9) / (ms / 1e3));
        printf("  Est. GFLOP/s: %.2f\n", gflops);
        printf("  Est. Bandwidth: %.2f GB/s\n", bandwidth);
        printf("  Verification: %s\n\n", ok ? "PASSED" : "FAILED");
    }

    if (runAll || strcmp(mode, "coarsened") == 0)
    {
        ms = run_and_time(d_in, d_out, nx, ny, nz, launch_coarsened, "stencil_coarsened");
        CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, dataSize, cudaMemcpyDeviceToHost));
        int ok = verify_stencil(h_out_gpu, h_out_cpu, nx, ny, nz, tolerance);

        double gflops = (interiorPoints * 13.0 / 1e9) / (ms / 1e3);
        double bandwidth = (interiorPoints * 7 * sizeof(float) + interiorPoints * sizeof(float)) / 1e9 / (ms / 1e3);

        printf("Kernel: stencil_coarsened (tile %dx%d, coarse %d)\n", TILE_X, TILE_Y, COARSE_Z);
        printf("  Time: %.3f ms\n", ms);
        printf("  Throughput: %.2f GPoints/s\n", (interiorPoints / 1e9) / (ms / 1e3));
        printf("  Est. GFLOP/s: %.2f\n", gflops);
        printf("  Est. Bandwidth: %.2f GB/s\n", bandwidth);
        printf("  Verification: %s\n\n", ok ? "PASSED" : "FAILED");
    }

    if (runAll || strcmp(mode, "register") == 0)
    {
        ms = run_and_time(d_in, d_out, nx, ny, nz, launch_register, "stencil_register");
        CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, dataSize, cudaMemcpyDeviceToHost));
        int ok = verify_stencil(h_out_gpu, h_out_cpu, nx, ny, nz, tolerance);

        double gflops = (interiorPoints * 13.0 / 1e9) / (ms / 1e3);
        double bandwidth = (interiorPoints * 7 * sizeof(float) + interiorPoints * sizeof(float)) / 1e9 / (ms / 1e3);

        printf("Kernel: stencil_register (tile %dx%d, register z-sweep)\n", TILE_X, TILE_Y);
        printf("  Time: %.3f ms\n", ms);
        printf("  Throughput: %.2f GPoints/s\n", (interiorPoints / 1e9) / (ms / 1e3));
        printf("  Est. GFLOP/s: %.2f\n", gflops);
        printf("  Est. Bandwidth: %.2f GB/s\n", bandwidth);
        printf("  Verification: %s\n\n", ok ? "PASSED" : "FAILED");
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out_gpu);
    free(h_out_cpu);

    return 0;
}
