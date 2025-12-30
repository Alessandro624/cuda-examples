# Error Handling Demos

This directory contains small programs that intentionally trigger common CUDA runtime and kernel errors so you can observe runtime messages and test profiling/debugging tools.

## Build

```bash
cd error_handling
make
```

## Usage

### vectAdd_errors

Vector-add demo with intentional error modes:

```bash
./vectAdd_errors [--mode MODE] [--n N]
```

**Modes:**

| Mode | Description |
|------|-------------|
| 0 | Safe run (no errors) |
| 1 | Excessive block size (invalid launch configuration) |
| 2 | Invalid host pointer passed to `cudaMemcpy` |
| 3 | Excessive allocation request (forced `cudaMalloc` failure) |
| 4 | Referencing invalid device pointer in kernel |
| 5 | Out-of-bounds global memory write in kernel |

### errorCudaMemcpy

Demonstrates common `cudaMemcpy` and memory-management mistakes:
- Incorrect sizes
- nullptr copies
- Misuse of `cudaMemcpyDeviceToDevice`

## Run

```bash
# Run vectAdd_errors in a specific mode
./run.sh vectAdd_errors --mode 1 --n 1024

# Run the errorCudaMemcpy demo
./run.sh errorCudaMemcpy --n 1024
```

## Profiling

```bash
./profile_nvprof.sh errorCudaMemcpy
```

## Notes

- These examples are **intentionally invalid** — run them in a controlled environment for learning and debugging.
- The programs print CUDA error strings produced by the runtime.
- Use `nvprof` or `compute-sanitizer` to inspect kernel activity and memory events.
