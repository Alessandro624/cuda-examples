# Convolution Examples

CUDA implementations of 1D and 2D convolution with various optimization strategies including constant memory and shared memory tiling.

## Build

```bash
cd convolution
make
```

This builds two executables:
- `convolution1D` - 1D convolution examples
- `convolution2D` - 2D convolution examples

## Programs and Usage

### convolution1D

1D convolution with constant memory filter and optional tiling.

```bash
./convolution1D [--n N]
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--n` | Length of the 1D signal | 1000000 |

**Kernels executed:**
- `convolution1DWithConstantMemoryKernel` - Uses constant memory for filter coefficients
- `convolution1DWithCacheAndTilingKernel` - Adds shared memory tiling for input data

**Example:**
```bash
./convolution1D --n 2000000
```

### convolution2D

2D convolution with multiple optimization levels.

```bash
./convolution2D [--width W] [--height H]
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--width` | Width of the 2D image | 1920 |
| `--height` | Height of the 2D image | 1080 |

**Kernels executed:**
- `convolution2DBasicKernel` - Naive implementation with global memory filter
- `convolution2DWithConstantMemoryKernel` - Filter stored in constant memory
- `convolution2DWithTilingKernel` - Input/output tiling with shared memory
- `convolution2DWithCacheAndTilingKernel` - Combined caching and tiling optimization

**Example:**
```bash
./convolution2D --width 1920 --height 1080
```

## Run Examples

Use the runner script:

```bash
# 1D convolution
./run.sh 1D --n 5000000

# 2D convolution
./run.sh 2D --width 3840 --height 2160
```

## Profile with nvprof

```bash
# Profile 1D convolution
./profile_nvprof.sh 1D --n 1000000

# Profile 2D convolution
./profile_nvprof.sh 2D --width 1920 --height 1080
```

## Implementation Details

### Filter Parameters

Both implementations use a compile-time filter radius defined as:
- `FILTER_RADIUS = 9` → Filter size: 19×1 (1D) or 19×19 (2D)

### Memory Hierarchy

| Kernel Type | Filter Storage | Input Caching |
|-------------|----------------|---------------|
| Basic | Global memory | None |
| Constant Memory | Constant memory | None |
| Tiling | Constant memory | Shared memory |
| Cache + Tiling | Constant memory | Shared memory with halo |

### Tile Dimensions

**1D Convolution:**
- `TILE_DIM = 64` threads per block

**2D Convolution:**
- `TILE_DIM = 32` for cache+tiling kernel
- `IN_TILE_DIM = 32`, `OUT_TILE_DIM = 14` for pure tiling kernel (accounts for halo)

## Notes

- Constant memory provides broadcast capability for filter coefficients accessed by all threads.
- Tiling reduces global memory bandwidth by reusing data in shared memory.
- The halo region in tiled implementations handles boundary conditions.
- Use `NVCCFLAGS` in the Makefile to tune compilation flags for your hardware.
- Use profiling tools: `../profiling_tools/profile_cuda.sh -d .`
