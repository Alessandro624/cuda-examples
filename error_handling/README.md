# Error-handling demos for vector addition

This directory contains small programs that intentionally trigger common CUDA
runtime and kernel errors so you can observe runtime messages and test
profiling/debugging tools such as `nvprof`.

Build:

```bash
cd error_handling
make
```

Programs and usage:

- `vectAdd_errors [--mode idx]` : vector-add demo with intentional error modes
  - Mode 0 (or no mode): safe run
  - Mode 1 : excessive block size (invalid kernel launch configuration)
  - Mode 2 : invalid host pointer passed to `cudaMemcpy`
  - Mode 3 : excessive allocation request (forced `cudaMalloc` failure)
  - Mode 4 : referencing invalid device pointer in kernel (NULL/invalid)
  - Mode 5 : out-of-bounds global memory write in kernel

- `errorCudaMemcpy` : separate demo that demonstrates common cudaMemcpy /
  memory-management mistakes, including incorrect sizes, nullptr copies and
  misuse of `cudaMemcpyDeviceToDevice`. This file includes its own checking
  macros and intentionally triggers runtime/runtime-sticky errors for testing.

Run examples (local runner):

```bash
# Run vectAdd_errors in a specific mode:
./run.sh vectAdd_errors --mode 1 --n 1024
# Run the errorCudaMemcpy demo:
./run.sh errorCudaMemcpy --n 1024
```

Profile with nvprof:

```bash
./profile_nvprof.sh errorCudaMemcpy
```

Notes:

- The examples are intentionally invalid — run them in a controlled environment
  for learning and debugging. The programs print CUDA error strings produced by
  the runtime. Use `nvprof` output to inspect kernel activity and memory events.
