# Matrix Multiplication

Multiple kernel implementations for matrix-matrix multiplication demonstrating different optimization strategies: naive, tiled (shared memory), coarsened, and per-row/per-column variants.

## Build

```bash
cd matrix_multiplication
make
```

## Usage

```bash
./matrixMul [--mode MODE] [--M M] [--K K] [--N N] [--threads T] [--tile TILE] [--coarse C]
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Kernel: `naive`, `tiled`, `coarsened`, `perrows`, `percols`, `all` | `all` |
| `--M` | Matrix A rows | 1024 |
| `--K` | Matrix A cols / B rows | 1024 |
| `--N` | Matrix B cols | 1024 |
| `--threads` | Threads per block | 256 |
| `--tile` | Tile dimension for shared memory | 16 |
| `--coarse` | Coarsening factor (1-8) | 2 |

## Run

```bash
./run.sh --mode=tiled --M=1024 --K=1024 --N=1024 --threads=256 --tile=16
```

## Profiling

```bash
# Profile with nvprof
./profile_nvprof.sh --M 1024 --K 1024 --N 1024 --threads 256

# Use profiling tools
../profiling_tools/profile_cuda.sh -d .
```

## Notes

- The CUDA kernels demonstrate multiple implementation strategies for microbenchmarking.
- Use the profiling scripts and gnuplot to collect timings and generate Roofline / bar charts.
- Tiled kernel uses shared memory to reduce global memory bandwidth requirements.
- Coarsened kernel computes multiple output elements per thread.
