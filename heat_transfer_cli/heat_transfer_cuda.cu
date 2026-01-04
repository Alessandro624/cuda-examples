#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ============================================================================
// CUDA Error Handling
// ============================================================================
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ============================================================================
// Timer Class
// ============================================================================
class Timer
{
private:
    cudaEvent_t start_, stop_;

public:
    Timer()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~Timer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop()
    {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ============================================================================
// Kernel Mode Enum
// ============================================================================
enum KernelMode
{
    GLOBAL = 0,
    TILED = 1,
    TILED_WH = 2
};

// ============================================================================
// Kernel 1: GLOBAL - Straightforward parallelization with Unified Memory
// ============================================================================
__global__ void updateTemperatureGlobal(double *TNext, const double *T,
                                        unsigned int nRows, unsigned int nCols,
                                        unsigned int rowStart, unsigned int rowEnd,
                                        unsigned int colStart, unsigned int colEnd)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + colStart;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y + rowStart;

    if (i <= rowEnd && j <= colEnd)
    {
        TNext[i * nCols + j] = (1.0 / 20.0) *
                               (4.0 * (T[i * nCols + (j + 1)] +
                                       T[i * nCols + (j - 1)] +
                                       T[(i + 1) * nCols + j] +
                                       T[(i - 1) * nCols + j]) +
                                T[(i + 1) * nCols + (j + 1)] +
                                T[(i + 1) * nCols + (j - 1)] +
                                T[(i - 1) * nCols + (j + 1)] +
                                T[(i - 1) * nCols + (j - 1)]);
    }
}

// ============================================================================
// Kernel 2: TILED - Shared memory tiled parallelization without halo cells
// Each thread loads one element to shared memory, but boundary threads
// read halo directly from global memory
// ============================================================================
template <int BLOCK_X, int BLOCK_Y>
__global__ void updateTemperatureTiled(double *TNext, const double *T,
                                       unsigned int nRows, unsigned int nCols,
                                       unsigned int rowStart, unsigned int rowEnd,
                                       unsigned int colStart, unsigned int colEnd)
{
    __shared__ double tile[BLOCK_Y][BLOCK_X];

    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + colStart;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y + rowStart;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (i <= rowEnd && j <= colEnd)
    {
        tile[ty][tx] = T[i * nCols + j];
    }
    __syncthreads();

    if (i <= rowEnd && j <= colEnd)
    {
        double east = (tx < BLOCK_X - 1 && j < colEnd) ? tile[ty][tx + 1] : T[i * nCols + (j + 1)];
        double west = (tx > 0 && j > colStart) ? tile[ty][tx - 1] : T[i * nCols + (j - 1)];
        double south = (ty < BLOCK_Y - 1 && i < rowEnd) ? tile[ty + 1][tx] : T[(i + 1) * nCols + j];
        double north = (ty > 0 && i > rowStart) ? tile[ty - 1][tx] : T[(i - 1) * nCols + j];

        double se = T[(i + 1) * nCols + (j + 1)];
        double sw = T[(i + 1) * nCols + (j - 1)];
        double ne = T[(i - 1) * nCols + (j + 1)];
        double nw = T[(i - 1) * nCols + (j - 1)];

        TNext[i * nCols + j] = (1.0 / 20.0) *
                               (4.0 * (east + west + south + north) + se + sw + ne + nw);
    }
}

// ============================================================================
// Kernel 3: TILED_WH - Tiled parallelization with halo cells in shared memory
// Block's boundary threads perform extra work to copy halo cells
// ============================================================================
template <int BLOCK_X, int BLOCK_Y>
__global__ void updateTemperatureTiledWithHalo(double *TNext, const double *T,
                                               unsigned int nRows, unsigned int nCols,
                                               unsigned int rowStart, unsigned int rowEnd,
                                               unsigned int colStart, unsigned int colEnd)
{
    __shared__ double tile[BLOCK_Y + 2][BLOCK_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * BLOCK_X + tx + colStart;
    int i = blockIdx.y * BLOCK_Y + ty + rowStart;
    int sx = tx + 1;
    int sy = ty + 1;

    // Initialize tile to zero (for safety at boundaries)
    tile[sy][sx] = 0.0;
    if (ty == 0)
        tile[0][sx] = 0.0;
    if (ty == BLOCK_Y - 1)
        tile[BLOCK_Y + 1][sx] = 0.0;
    if (tx == 0)
        tile[sy][0] = 0.0;
    if (tx == BLOCK_X - 1)
        tile[sy][BLOCK_X + 1] = 0.0;
    if (tx == 0 && ty == 0)
        tile[0][0] = 0.0;
    if (tx == BLOCK_X - 1 && ty == 0)
        tile[0][BLOCK_X + 1] = 0.0;
    if (tx == 0 && ty == BLOCK_Y - 1)
        tile[BLOCK_Y + 1][0] = 0.0;
    if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1)
        tile[BLOCK_Y + 1][BLOCK_X + 1] = 0.0;

    // Load interior cell (if within valid range)
    if (i >= 0 && i < (int)nRows && j >= 0 && j < (int)nCols)
    {
        tile[sy][sx] = T[i * nCols + j];
    }

    // Load halo cells - boundary threads do extra work
    if (ty == 0)
    {
        int gi = i - 1;
        if (gi >= 0 && j >= 0 && j < (int)nCols)
        {
            tile[0][sx] = T[gi * nCols + j];
        }
    }
    if (ty == BLOCK_Y - 1)
    {
        int gi = i + 1;
        if (gi < (int)nRows && j >= 0 && j < (int)nCols)
        {
            tile[BLOCK_Y + 1][sx] = T[gi * nCols + j];
        }
    }

    if (tx == 0)
    {
        int gj = j - 1;
        if (gj >= 0 && i >= 0 && i < (int)nRows)
        {
            tile[sy][0] = T[i * nCols + gj];
        }
    }
    if (tx == BLOCK_X - 1)
    {
        int gj = j + 1;
        if (gj < (int)nCols && i >= 0 && i < (int)nRows)
        {
            tile[sy][BLOCK_X + 1] = T[i * nCols + gj];
        }
    }

    // Corner halos
    if (tx == 0 && ty == 0)
    {
        int gi = i - 1, gj = j - 1;
        if (gi >= 0 && gj >= 0)
        {
            tile[0][0] = T[gi * nCols + gj];
        }
    }
    if (tx == BLOCK_X - 1 && ty == 0)
    {
        int gi = i - 1, gj = j + 1;
        if (gi >= 0 && gj < (int)nCols)
        {
            tile[0][BLOCK_X + 1] = T[gi * nCols + gj];
        }
    }
    if (tx == 0 && ty == BLOCK_Y - 1)
    {
        int gi = i + 1, gj = j - 1;
        if (gi < (int)nRows && gj >= 0)
        {
            tile[BLOCK_Y + 1][0] = T[gi * nCols + gj];
        }
    }
    if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1)
    {
        int gi = i + 1, gj = j + 1;
        if (gi < (int)nRows && gj < (int)nCols)
        {
            tile[BLOCK_Y + 1][BLOCK_X + 1] = T[gi * nCols + gj];
        }
    }

    __syncthreads();

    if (i >= (int)rowStart && i <= (int)rowEnd && j >= (int)colStart && j <= (int)colEnd)
    {
        double east = tile[sy][sx + 1];
        double west = tile[sy][sx - 1];
        double south = tile[sy + 1][sx];
        double north = tile[sy - 1][sx];
        double se = tile[sy + 1][sx + 1];
        double sw = tile[sy + 1][sx - 1];
        double ne = tile[sy - 1][sx + 1];
        double nw = tile[sy - 1][sx - 1];

        TNext[i * nCols + j] = (1.0 / 20.0) *
                               (4.0 * (east + west + south + north) + se + sw + ne + nw);
    }
}

// ============================================================================
// I/O Functions
// ============================================================================
void saveTemperature(const std::string &fileBaseName, const std::string &fileExtension,
                     unsigned int step, const double *gridTemperature,
                     unsigned int nRows, unsigned int nCols, unsigned int fieldW)
{
    std::string filename = fileBaseName + "_step_" + std::to_string(step) + fileExtension;
    std::ofstream file(filename);
    if (file.is_open())
    {
        for (unsigned int i = 0; i < nRows; i++)
        {
            for (unsigned int j = 0; j < nCols; j++)
                file << std::setw(fieldW) << gridTemperature[i * nCols + j] << " ";
            file << std::endl;
        }
        file << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file " << filename << "!" << std::endl;
    }
    file.close();
}

// ============================================================================
// Initialization Function
// ============================================================================
void initTopBottomTemperature(double *gridTemperature, unsigned int nRows, unsigned int nCols,
                              unsigned int nTopRows, unsigned int nBottomRows, double temperature)
{
    for (unsigned int i = 0; i < nTopRows; i++)
        for (unsigned int j = 0; j < nCols; j++)
            gridTemperature[i * nCols + j] = temperature;

    for (unsigned int i = nTopRows; i < nRows - nBottomRows; i++)
        for (unsigned int j = 0; j < nCols; j++)
            gridTemperature[i * nCols + j] = 0.0;

    for (unsigned int i = nRows - nBottomRows; i < nRows; i++)
        for (unsigned int j = 0; j < nCols; j++)
            gridTemperature[i * nCols + j] = temperature;
}

// ============================================================================
// Print Usage
// ============================================================================
void printUsage(const char *progName)
{
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --help, -h         Show this help message\n"
              << "  --mode M           Kernel mode: 0=Global, 1=Tiled, 2=Tiled_wH (default: 0)\n"
              << "  --block BX BY      Block dimensions (default: 16 16)\n"
              << "                     Supported: 8x8, 8x16, 8x32, 16x8, 16x16, 16x32, 32x8, 32x16, 32x32\n"
              << "  --steps N          Number of simulation steps (default: 10000)\n"
              << "  --rows R           Grid rows (default: 256)\n"
              << "  --cols C           Grid columns (default: 4096)\n"
              << "  --hotrows T B      Hot rows at top and bottom (default: 2 2)\n"
              << "  --temp T           Initial hot temperature (default: 20.0)\n"
              << "  --save             Save initial and final configurations\n"
              << "  --verify           Verify result against CPU reference\n"
              << std::endl;
}

// ============================================================================
// Kernel Launcher
// ============================================================================
void launchKernel(KernelMode mode, dim3 grid, dim3 block,
                  double *TNext, const double *T,
                  unsigned int nRows, unsigned int nCols,
                  unsigned int rowStart, unsigned int rowEnd,
                  unsigned int colStart, unsigned int colEnd)
{

    int bx = block.x;
    int by = block.y;

    switch (mode)
    {
    case GLOBAL:
        updateTemperatureGlobal<<<grid, block>>>(TNext, T, nRows, nCols,
                                                 rowStart, rowEnd, colStart, colEnd);
        break;
    case TILED:
        // Select template based on block size
        if (bx == 8 && by == 8)
            updateTemperatureTiled<8, 8><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 16 && by == 8)
            updateTemperatureTiled<16, 8><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 32 && by == 8)
            updateTemperatureTiled<32, 8><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 8 && by == 16)
            updateTemperatureTiled<8, 16><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 16 && by == 16)
            updateTemperatureTiled<16, 16><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 32 && by == 16)
            updateTemperatureTiled<32, 16><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 8 && by == 32)
            updateTemperatureTiled<8, 32><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 16 && by == 32)
            updateTemperatureTiled<16, 32><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 32 && by == 32)
            updateTemperatureTiled<32, 32><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else
        {
            std::cerr << "Unsupported block size for TILED: " << bx << "x" << by << std::endl;
            exit(EXIT_FAILURE);
        }
        break;
    case TILED_WH:
        if (bx == 8 && by == 8)
            updateTemperatureTiledWithHalo<8, 8><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 16 && by == 8)
            updateTemperatureTiledWithHalo<16, 8><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 32 && by == 8)
            updateTemperatureTiledWithHalo<32, 8><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 8 && by == 16)
            updateTemperatureTiledWithHalo<8, 16><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 16 && by == 16)
            updateTemperatureTiledWithHalo<16, 16><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 32 && by == 16)
            updateTemperatureTiledWithHalo<32, 16><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 8 && by == 32)
            updateTemperatureTiledWithHalo<8, 32><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 16 && by == 32)
            updateTemperatureTiledWithHalo<16, 32><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else if (bx == 32 && by == 32)
            updateTemperatureTiledWithHalo<32, 32><<<grid, block>>>(TNext, T, nRows, nCols, rowStart, rowEnd, colStart, colEnd);
        else
        {
            std::cerr << "Unsupported block size for TILED_WH: " << bx << "x" << by << std::endl;
            exit(EXIT_FAILURE);
        }
        break;
    }
}

// ============================================================================
// CPU Reference for Verification
// ============================================================================
void cpuUpdateRegion(double *next, const double *curr,
                     unsigned int nRows, unsigned int nCols,
                     unsigned int rowStart, unsigned int rowEnd,
                     unsigned int colStart, unsigned int colEnd)
{
    for (unsigned int i = rowStart; i <= rowEnd; i++)
    {
        for (unsigned int j = colStart; j <= colEnd; j++)
        {
            next[i * nCols + j] = (1.0 / 20.0) *
                                  (4.0 * (curr[i * nCols + (j + 1)] +
                                          curr[i * nCols + (j - 1)] +
                                          curr[(i + 1) * nCols + j] +
                                          curr[(i - 1) * nCols + j]) +
                                   curr[(i + 1) * nCols + (j + 1)] +
                                   curr[(i + 1) * nCols + (j - 1)] +
                                   curr[(i - 1) * nCols + (j + 1)] +
                                   curr[(i - 1) * nCols + (j - 1)]);
        }
    }
}

bool verifyResult(const double *gpu, const double *cpu, unsigned int size, double tolerance = 1e-6)
{
    for (unsigned int i = 0; i < size; i++)
    {
        if (std::abs(gpu[i] - cpu[i]) > tolerance)
        {
            std::cerr << "Mismatch at index " << i << ": GPU=" << gpu[i] << " CPU=" << cpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char *argv[])
{
    // Default parameters
    unsigned int nSteps = 10000;
    unsigned int gridRows = 256;
    unsigned int gridCols = 4096;
    unsigned int nHotTopRows = 2;
    unsigned int nHotBottomRows = 2;
    double initialHotTemperature = 20.0;
    unsigned int fieldWidth = 5;
    std::string outfilePrefix = "temperature";
    std::string outfileExtension = ".dat";

    KernelMode mode = GLOBAL;
    unsigned int blockX = 16, blockY = 16;
    bool saveOutput = false;
    bool verify = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
        {
            mode = static_cast<KernelMode>(atoi(argv[++i]));
        }
        else if (strcmp(argv[i], "--block") == 0 && i + 2 < argc)
        {
            blockX = atoi(argv[++i]);
            blockY = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
        {
            nSteps = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc)
        {
            gridRows = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--cols") == 0 && i + 1 < argc)
        {
            gridCols = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--hotrows") == 0 && i + 2 < argc)
        {
            nHotTopRows = atoi(argv[++i]);
            nHotBottomRows = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc)
        {
            initialHotTemperature = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--save") == 0)
        {
            saveOutput = true;
        }
        else if (strcmp(argv[i], "--verify") == 0)
        {
            verify = true;
        }
    }

    // Print configuration
    const char *modeNames[] = {"Global", "Tiled", "Tiled_wH"};
    std::cout << "========================================" << std::endl;
    std::cout << "Heat Transfer CUDA Simulation" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Mode:          " << modeNames[mode] << std::endl;
    std::cout << "Block size:    " << blockX << "x" << blockY << std::endl;
    std::cout << "Grid size:     " << gridRows << "x" << gridCols << std::endl;
    std::cout << "Steps:         " << nSteps << std::endl;
    std::cout << "Hot rows:      top=" << nHotTopRows << ", bottom=" << nHotBottomRows << std::endl;
    std::cout << "Temperature:   " << initialHotTemperature << std::endl;
    std::cout << "========================================" << std::endl;

    // Calculate grid dimensions
    unsigned int rowStart = nHotTopRows;
    unsigned int rowEnd = gridRows - 1 - nHotBottomRows;
    unsigned int colStart = 1;
    unsigned int colEnd = gridCols - 2;

    unsigned int computeRows = rowEnd - rowStart + 1;
    unsigned int computeCols = colEnd - colStart + 1;

    // Allocate unified memory
    double *temperatureCurrent = nullptr;
    double *temperatureNext = nullptr;
    size_t size = gridRows * gridCols * sizeof(double);

    CUDA_CHECK(cudaMallocManaged(&temperatureCurrent, size));
    CUDA_CHECK(cudaMallocManaged(&temperatureNext, size));

    // Initialize temperature
    initTopBottomTemperature(temperatureCurrent, gridRows, gridCols,
                             nHotTopRows, nHotBottomRows, initialHotTemperature);
    initTopBottomTemperature(temperatureNext, gridRows, gridCols,
                             nHotTopRows, nHotBottomRows, initialHotTemperature);

    // Save initial configuration
    if (saveOutput)
    {
        std::cout << "Saving initial configuration... " << std::endl;
        saveTemperature(outfilePrefix, outfileExtension, 0, temperatureCurrent,
                        gridRows, gridCols, fieldWidth);
        std::cout << "Done" << std::endl;
    }

    // Set up CUDA grid and block dimensions
    dim3 block(blockX, blockY);
    dim3 grid((computeCols + blockX - 1) / blockX,
              (computeRows + blockY - 1) / blockY);

    std::cout << "CUDA grid:     " << grid.x << "x" << grid.y << " blocks" << std::endl;
    std::cout << "CUDA block:    " << block.x << "x" << block.y << " threads" << std::endl;
    std::cout << "========================================" << std::endl;

    // Run simulation
    std::cout << "Simulation in progress... " << std::endl;
    Timer timer;
    timer.start();

    for (unsigned int step = 1; step <= nSteps; step++)
    {
        launchKernel(mode, grid, block, temperatureNext, temperatureCurrent,
                     gridRows, gridCols, rowStart, rowEnd, colStart, colEnd);

        // Swap pointers
        double *temp = temperatureNext;
        temperatureNext = temperatureCurrent;
        temperatureCurrent = temp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    float elapsedTime = timer.stop();

    std::cout << "Simulation loop elapsed time: " << elapsedTime << " ms "
              << "(corresponding to " << (elapsedTime / 1000.0) << " s)" << std::endl;

    // Verification against CPU
    if (verify)
    {
        std::cout << "Verifying against CPU reference (running " << nSteps << " steps)..." << std::endl;

        double *cpuCurrent = new double[gridRows * gridCols];
        double *cpuNext = new double[gridRows * gridCols];

        initTopBottomTemperature(cpuCurrent, gridRows, gridCols,
                                 nHotTopRows, nHotBottomRows, initialHotTemperature);
        initTopBottomTemperature(cpuNext, gridRows, gridCols,
                                 nHotTopRows, nHotBottomRows, initialHotTemperature);

        for (unsigned int step = 1; step <= nSteps; step++)
        {
            cpuUpdateRegion(cpuNext, cpuCurrent, gridRows, gridCols,
                            rowStart, rowEnd, colStart, colEnd);
            double *temp = cpuNext;
            cpuNext = cpuCurrent;
            cpuCurrent = temp;
        }

        if (verifyResult(temperatureCurrent, cpuCurrent, gridRows * gridCols))
        {
            std::cout << "Verification PASSED!" << std::endl;
        }
        else
        {
            std::cout << "Verification FAILED!" << std::endl;
        }

        delete[] cpuCurrent;
        delete[] cpuNext;
    }

    // Save final configuration
    if (saveOutput)
    {
        std::cout << "Saving final configuration... " << std::endl;
        saveTemperature(outfilePrefix, outfileExtension, nSteps, temperatureCurrent,
                        gridRows, gridCols, fieldWidth);
        std::cout << "Done" << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(temperatureCurrent));
    CUDA_CHECK(cudaFree(temperatureNext));

    return 0;
}
