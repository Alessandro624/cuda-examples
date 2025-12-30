# 3D Seven-Point Stencil

## Overview

This example demonstrates multiple CUDA implementations of the 3D seven-point stencil, a fundamental computational pattern in scientific computing used for solving PDEs, heat diffusion, and iterative solvers.

The seven-point stencil computes:

```
out[i,j,k] = c0*in[i,j,k]   + c1*in[i-1,j,k] + c2*in[i+1,j,k] +
             c3*in[i,j-1,k] + c4*in[i,j+1,k] +
             c5*in[i,j,k-1] + c6*in[i,j,k+1]
```

Where coefficients are defined as (discrete Laplacian):

- `c0 = -6.0` (center)
- `c1 = c2 = c3 = c4 = c5 = c6 = 1.0` (neighbors)

## Kernel Implementations

### 1. Naive (`stencil_naive`)
- **Strategy**: Direct global memory access
- **Characteristics**:
  - Each thread computes one output point
  - 3D thread block organization (8×8×8)
  - Simple but memory bandwidth limited
  - Redundant loads from global memory

### 2. Shared Memory Tiling (`stencil_shared`)
- **Strategy**: 2D xy-plane tiling with shared memory
- **Characteristics**:
  - Loads xy-plane tiles into shared memory (with halo)
  - Reduces redundant global memory accesses for xy-neighbors
  - z-neighbors still loaded from global memory
  - One block per z-layer

### 3. Thread Coarsening (`stencil_coarsened`)
- **Strategy**: Each thread processes multiple z-layers
- **Characteristics**:
  - Combines shared memory tiling with z-axis coarsening
  - Reduces thread launch overhead
  - Better data reuse along z-dimension
  - Configurable coarsening factor (default: 8)

### 4. Register Tiling (`stencil_register`)
- **Strategy**: Register caching along z-axis
- **Characteristics**:
  - Maintains sliding window of z-values in registers
  - Maximizes temporal reuse along z-dimension
  - Combines with xy-plane shared memory tiling
  - Most efficient memory access pattern

## Build

```bash
make
```

## Run

```bash
# Show help
./run.sh --help

# Run all kernels (default 256³ grid)
./run.sh --mode all

# Run specific kernel
./run.sh --mode naive --nx 128 --ny 128 --nz 128
./run.sh --mode shared --nx 512 --ny 512 --nz 256
./run.sh --mode coarsened
./run.sh --mode register
```

## Profiling

```bash
# Profile with nvprof
./profile_nvprof.sh --mode all --nx 256 --ny 256 --nz 256

# Use with profiling tools
../profiling_tools/profile_cuda.sh -d . --metrics time,occupancy
```

## Output Example

```
=== 3D Seven-Point Stencil ===
Grid: 256 x 256 x 256 = 16777216 elements
Interior points: 16003008
Data size: 64.00 MB
Mode: all

Computing CPU reference...

Kernel: stencil_naive
  Time: 12.345 ms
  Throughput: 1.30 GPoints/s
  Est. GFLOP/s: 16.87
  Est. Bandwidth: 58.23 GB/s
  Verification: PASSED

Kernel: stencil_shared (tile 32x8)
  Time: 8.234 ms
  Throughput: 1.94 GPoints/s
  Est. GFLOP/s: 25.26
  Est. Bandwidth: 87.12 GB/s
  Verification: PASSED

Kernel: stencil_coarsened (tile 32x8, coarse 8)
  Time: 6.123 ms
  Throughput: 2.61 GPoints/s
  Est. GFLOP/s: 33.96
  Est. Bandwidth: 117.11 GB/s
  Verification: PASSED

Kernel: stencil_register (tile 32x8, register z-sweep)
  Time: 4.567 ms
  Throughput: 3.50 GPoints/s
  Est. GFLOP/s: 45.55
  Est. Bandwidth: 157.12 GB/s
  Verification: PASSED
```

## Optimization Techniques

### 1. Shared Memory Tiling
- Reduces redundant global memory accesses
- Threads in a block cooperatively load data once
- Halo regions handle boundary conditions

### 2. Thread Coarsening
- Amortizes thread launch overhead
- Better instruction-level parallelism
- Improves data reuse within a thread

### 3. Register Tiling
- Exploits temporal locality along sweep direction
- Registers provide fastest memory access
- Sliding window technique minimizes register pressure

### 4. Memory Coalescing
- Row-major layout ensures coalesced accesses in x-direction
- Tile dimensions chosen for optimal memory access patterns

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `all` | Kernel selection |
| `--nx` | 256 | Grid size in X |
| `--ny` | 256 | Grid size in Y |
| `--nz` | 256 | Grid size in Z |

## Tile Configuration

- **Shared/Coarsened/Register**: 32×8 xy-tile (256 threads)
- **Coarsening factor**: 8 z-layers per thread
- **Shared memory**: (32+2)×(8+2)×4 = 1360 bytes per block

## Performance Considerations

1. **Grid Size**: Larger grids improve GPU utilization
2. **Memory Bandwidth**: Stencil kernels are typically memory-bound
3. **Occupancy**: Tile size affects SM occupancy
4. **Register Pressure**: Register tiling may limit occupancy

## Notes

- Uses 7 independent coefficients (c0-c6) for flexibility.
- Host-side verification ensures correctness against CPU reference.
- All kernels include boundary handling (skip boundary points).
- Use `NVCCFLAGS` in the `Makefile` to tune compilation flags.
