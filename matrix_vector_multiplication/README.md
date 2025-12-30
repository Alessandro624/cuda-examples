# Matrix-Vector Multiplication

Simple example that multiplies a matrix A (height × width) by a vector B (width) producing vector C (height) using a straightforward GPU kernel.

## Build

```bash
cd matrix_vector_multiplication
make
```

## Usage

```bash
./matrixVectMul [--width W] [--height H] [--threads T]
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--width` | Matrix width / vector length | 1024 |
| `--height` | Matrix height / output length | 1024 |
| `--threads` | Threads per block | 256 |

## Run

```bash
./run.sh --width=2048 --height=1024 --threads=128
```

## Profiling

```bash
# Profile with nvprof
./profile_nvprof.sh --width=2048 --height=1024

# Use profiling tools
../profiling_tools/profile_cuda.sh -d .
```

## Notes

- This implementation demonstrates per-row parallelization where each thread computes one output element.
- Intentionally simple; does not use shared-memory tiling or other optimizations.
- Use `NVCCFLAGS` in the `Makefile` to tune compile flags.
