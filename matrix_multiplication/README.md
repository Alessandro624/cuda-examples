# matrix_multiplication

Examples and microbenchmarks for matrix-matrix multiplication. This subproject includes
multiple kernel variants (naive, tiled/shared-memory, coarsened and per-row/col variants),
convenience runner and profiling helpers.

**Build**

```bash
cd matrix_multiplication
make
```

**Programs and usage**

- `matrixMul [--mode MODE] [--M M] [--K K] [--N N] [--threads THREADS] [--tile TILE] [--coarse COARSE]` : Run a single mode.
- Modes (supported): `naive`, `tiled`, `coarsened`, `perrows`, `percols`.
	- `coarsened` accepts additional `COARSE` parameter (1..8) as last argument.

**Run (local runner)**

```bash
./run.sh --mode=tiled --M=1024 --K=1024 --N=1024 --threads=256 --tile=16
```

**Profile with nvprof**

```bash
./profile_nvprof.sh --M 1024 --K 1024 --N 1024 --threads 256
```

Notes

- The CUDA kernels intentionally demonstrate multiple implementation strategies for microbenchmarking; they are not heavily optimized for every GPU. Use the profiling scripts and gnuplot files to collect timings and generate a Roofline / bar chart.
