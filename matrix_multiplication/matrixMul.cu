/*
 * Matrix Multiplication in CUDA
 * Each thread computes one element of the output matrix C.
 * Several kernel implementations provided: naive, tiled (shared memory), coarsened, per-row, per-column.
 *
 * Usage:
 *  matrixMul [--mode MODE] [--M M] [--K K] [--N N] [--threads THREADS] [--tile TILE] [--coarse-factor COARSE_FACTOR]
 *
 * - mode:  naive|tiled|coarsened|perrows|percols (default: tiled)
 * - M,K,N: matrix dimensions (default: 2048 1024 512)
 * - THREADS: threads per block or similar (default: 256)
 * - TILE: tile width for tiled kernels (default: 16)
 * - COARSE_FACTOR: coarsening factor for coarsened kernel (default: 4)
 *
 * Host-side verification is included.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
#define M_DEFAULT (1 << 11)
#define K_DEFAULT (1 << 10)
#define N_DEFAULT (1 << 9)
#define THREADS_DEFAULT 256
#define TILE_WIDTH_DEFAULT 16
#define COARSE_FACTOR_DEFAULT 4
#define MAX_TILE 32

// GPU init kernel: set A to 0.5 * 2^-10, B to 2, C to 0
__global__ void init_matrices(float *A, float *B, float *C, int M, int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long totalA = (long)M * K;
    long totalB = (long)K * N;
    long totalC = (long)M * N;

    // initialize A
    for (long i = idx; i < totalA; i += blockDim.x * gridDim.x)
        A[i] = 0.5f * powf(2.0f, -10.0f);

    // initialize B
    for (long i = idx; i < totalB; i += blockDim.x * gridDim.x)
        B[i] = 2.0f;

    // initialize C
    for (long i = idx; i < totalC; i += blockDim.x * gridDim.x)
        C[i] = 0.0f;
}

// naive kernel: each thread computes one element C[row*N + col]
__global__ void matmul_naive(const float *A, const float *B, float *C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
    {
        float tmp = 0.0f;
        for (int k = 0; k < K; ++k)
            tmp += A[row * K + k] * B[k * N + col];
        C[row * N + col] = tmp;
    }
}

// tiled kernel (shared memory), TILE configurable at launch
template <int TILE>
__global__ void matmul_tiled(const float *A, const float *B, float *C, int M, int K, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        int aRow = row;
        int aCol = t * TILE + threadIdx.x;
        if (aRow < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int bRow = t * TILE + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

__global__ void matrixMulKernelWithCoarsening(const float *A, const float *B, float *C, int M, int K, int N, int tile_width, int coarse)
{
    // runtime tile_width supported up to MAX_TILE
    int tw = min(tile_width, MAX_TILE);
    int CF = min(coarse, 8); // limit coarsening to 8
    __shared__ float Ads[MAX_TILE][MAX_TILE];
    __shared__ float Bds[MAX_TILE][MAX_TILE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int colStart = bx * tw * CF + tx;
    int row = by * tw + ty;

    float Cvalue[8];
    for (int c = 0; c < CF; c++)
        Cvalue[c] = 0.0f;

    for (int ph = 0; ph < (int)ceil((float)K / tw); ph++)
    {
        // load A tile element
        if (row < M && (ph * tw + tx) < K)
            Ads[ty][tx] = A[row * K + ph * tw + tx];
        else
            Ads[ty][tx] = 0.0f;

        for (int c = 0; c < CF; c++)
        {
            int col = colStart + c * tw;
            if (col < N && (ph * tw + ty) < K)
                Bds[ty][tx] = B[(ph * tw + ty) * N + col];
            else
                Bds[ty][tx] = 0.0f;
            __syncthreads();
            for (int k = 0; k < tw; ++k)
            {
                Cvalue[c] += Ads[ty][k] * Bds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < CF; c++)
    {
        int col = colStart + c * tw;
        if (row < M && col < N)
            C[row * N + col] = Cvalue[c];
    }
}

__global__ void matrixMulKernelPerRows(const float *A, const float *B, float *C, int M, int K, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M)
    {
        for (int c = col; c < N; c += blockDim.x * gridDim.x)
        {
            float Cvalue = 0;
            for (int k = 0; k < K; ++k)
                Cvalue += A[row * K + k] * B[k * N + c];
            C[row * N + c] = Cvalue;
        }
    }
}

__global__ void matrixMulKernelPerCols(const float *A, const float *B, float *C, int M, int K, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < N)
    {
        for (int r = row; r < M; r += blockDim.y * gridDim.y)
        {
            float Cvalue = 0;
            for (int k = 0; k < K; ++k)
                Cvalue += A[r * K + k] * B[k * N + col];
            C[r * N + col] = Cvalue;
        }
    }
}

// Host-side verification (naive)
int verify_host(const float *A, const float *B, const float *C, int M, int K, int N)
{
    double tol = 1e-3;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += (double)A[i * K + k] * (double)B[k * N + j];
            double diff = fabs(sum - (double)C[i * N + j]);
            if (diff > tol * fabs(sum) + 1e-6)
            {
                fprintf(stderr, "Mismatch at %d,%d: host=%f device=%f diff=%g\n", i, j, (float)sum, C[i * N + j], diff);
                return 0;
            }
        }
    }
    return 1;
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

/**
 * main - parse flags, run the selected matrix multiplication kernel and verify results.
 */
int main(int argc, char **argv)
{
    if (cli_has_help(argc, argv))
    {
        fprintf(stderr, "Usage: %s [--mode MODE] [--M M] [--K K] [--N N] [--threads THREADS] [--tile TILE] [--coarse-factor COARSE_FACTOR]\n", argv[0]);
        fprintf(stderr, "  mode: naive|tiled|coarsened|perrows|percols (default: tiled)\n");
        fprintf(stderr, "  M,K,N: matrix dims (default: 2048 1024 512)\n");
        fprintf(stderr, "  THREADS: threads per block or similar (default: 256)\n");
        fprintf(stderr, "  TILE: tile width for tiled kernels (default: 16)\n");
        fprintf(stderr, "  COARSE_FACTOR: coarsening factor for coarsened kernel (default: 4)\n");
        return 0;
    }

    const char *mode = "tiled";
    int M = M_DEFAULT;
    int K = K_DEFAULT;
    int N = N_DEFAULT;
    int THREADS = THREADS_DEFAULT;
    int TILE = TILE_WIDTH_DEFAULT;
    int COARSE_FACTOR = COARSE_FACTOR_DEFAULT;

    const char *v = NULL;
    v = cli_find_flag_value(argc, argv, "mode");
    if (v)
        mode = v;
    v = cli_find_flag_value(argc, argv, "M");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid M '%s'\n", v);
            return 1;
        }
        M = atoi(v);
    }
    v = cli_find_flag_value(argc, argv, "K");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid K '%s'\n", v);
            return 1;
        }
        K = atoi(v);
    }
    v = cli_find_flag_value(argc, argv, "N");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid N '%s'\n", v);
            return 1;
        }
        N = atoi(v);
    }
    v = cli_find_flag_value(argc, argv, "threads");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid threads '%s'\n", v);
            return 1;
        }
        THREADS = atoi(v);
    }
    v = cli_find_flag_value(argc, argv, "tile");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid tile '%s'\n", v);
            return 1;
        }
        TILE = atoi(v);
    }
    v = cli_find_flag_value(argc, argv, "coarse-factor");
    if (v)
    {
        if (!is_positive_integer_str(v))
        {
            fprintf(stderr, "Invalid coarse-factor '%s'\n", v);
            return 1;
        }
        COARSE_FACTOR = atoi(v);
    }

    printf("Mode=%s M=%d K=%d N=%d THREADS=%d TILE=%d COARSE_FACTOR=%d\n", mode, M, K, N, THREADS, TILE, COARSE_FACTOR);

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *A_h = (float *)malloc(sizeA);
    float *B_h = (float *)malloc(sizeB);
    float *C_h = (float *)malloc(sizeC);
    if (!A_h || !B_h || !C_h)
    {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    float *A_d = NULL, *B_d = NULL, *C_d = NULL;
    CHECK_CUDA(cudaMalloc((void **)&A_d, sizeA));
    CHECK_CUDA(cudaMalloc((void **)&B_d, sizeB));
    CHECK_CUDA(cudaMalloc((void **)&C_d, sizeC));

    // init on GPU
    int init_threads = THREADS;
    init_threads = clamp_threads_to_device(init_threads);
    int init_blocks = compute_blocks_from_elements((long long)M * (long long)N, init_threads);

    init_matrices<<<init_blocks, init_threads>>>(A_d, B_d, C_d, M, K, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy to host for verification reference
    CHECK_CUDA(cudaMemcpy(A_h, A_d, sizeA, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B_h, B_d, sizeB, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    // verify init values (quick checks)
    float expectedA = 0.5f * powf(2.0f, -10.0f);
    if (A_h[0] != expectedA || B_h[0] != 2.0f || C_h[0] != 0.0f)
        fprintf(stderr, "Init check mismatch: A[0]=%f B[0]=%f C[0]=%f\n", A_h[0], B_h[0], C_h[0]);

    if (strcmp(mode, "naive") == 0)
    {
        int TILE_W = TILE;
        dim3 block(TILE_W, TILE_W);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        matmul_naive<<<grid, block>>>(A_d, B_d, C_d, M, K, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (strcmp(mode, "tiled") == 0)
    {
        int TILE_W = TILE;
        dim3 block(TILE_W, TILE_W);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        if (TILE_W == 8)
            matmul_tiled<8><<<grid, block>>>(A_d, B_d, C_d, M, K, N);
        else if (TILE_W == 16)
            matmul_tiled<16><<<grid, block>>>(A_d, B_d, C_d, M, K, N);
        else if (TILE_W == 32)
            matmul_tiled<32><<<grid, block>>>(A_d, B_d, C_d, M, K, N);
        else
        {
            fprintf(stderr, "Unsupported TILE %d (supported 8/16/32)\n", TILE_W);
            return 1;
        }

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (strcmp(mode, "coarsened") == 0)
    {
        const int TILE_W = TILE;
        const int COARSE = COARSE_FACTOR;
        dim3 block(TILE_W, TILE_W);
        // grid is in columns grouped by COARSE
        dim3 grid((N + TILE_W * COARSE - 1) / (TILE_W * COARSE), (M + TILE_W - 1) / TILE_W);

        matrixMulKernelWithCoarsening<<<grid, block>>>(A_d, B_d, C_d, M, K, N, TILE_W, COARSE);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (strcmp(mode, "perrows") == 0)
    {
        int TILE_W = TILE;
        dim3 block(TILE_W, TILE_W);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        matrixMulKernelPerRows<<<grid, block>>>(A_d, B_d, C_d, M, K, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else if (strcmp(mode, "percols") == 0)
    {
        int TILE_W = TILE;
        dim3 block(TILE_W, TILE_W);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        matrixMulKernelPerCols<<<grid, block>>>(A_d, B_d, C_d, M, K, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    else
    {
        fprintf(stderr, "Unknown mode '%s'\n", mode);
        return 1;
    }

    // copy result back
    CHECK_CUDA(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    // verify correctness
    int ok = verify_host(A_h, B_h, C_h, M, K, N);
    if (!ok)
        printf("Verification: FAIL\n");
    else
        printf("Verification: PASS\n");
    // cleanup
    free(A_h);
    free(B_h);
    free(C_h);
    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));

    return 0;
}
