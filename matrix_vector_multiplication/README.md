# Matrix-Vector Multiplication

Purpose: simple example that multiplies a matrix A (height x width) by a vector B (width) producing vector C (height) using a straightforward GPU kernel.

Build:

```bash
cd matrix_vector_multiplication
make
```

Programs and usage:

- `matrixVectMul [--width W] [--height H] [--threads T]` : run the example (defaults to 1024 x 1024 with 256 threads if no args provided).

Run examples (local runner):

```bash
./run.sh --width=2048 --height=1024 --threads=128
```

Profile with nvprof:

```bash
./profile_nvprof.sh --width=2048 --height=1024
```

Notes:

- This implementation is intentionally simple. It demonstrates a per-row parallelization where each thread computes one output element. It does not attempt shared-memory tiling or other optimizations.
- Use `NVCCFLAGS` in the `Makefile` to tune compile flags.
