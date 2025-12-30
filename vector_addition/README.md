# Vector Addition (vectAdd)

Small demos that implement vector addition on the GPU. Variants include a baseline implementation, a grid-stride kernel with a thread granularity parameter, and Unified Memory versions (with and without prefetch).

## Build

```bash
cd vector_addition
make
```

## Usage

```bash
./vectAdd [--mode M] [--n N] [--threads T] [--granularity G]
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Implementation variant (see source or `-h`) | 0 |
| `--n` | Number of elements | 1000000 |
| `--threads` | Threads per block | 256 |
| `--granularity` | Elements per thread (grid-stride) | 1 |

**Example:**
```bash
./vectAdd --mode=2 --n=1000000 --threads=256 --granularity=4
```

## Run

```bash
./run.sh --mode=2 --n=1000000 --threads=256 --granularity=4
```

## Profiling

```bash
# Profile with nvprof
./profile_nvprof.sh --mode=1

# Use profiling tools
../profiling_tools/profile_cuda.sh -d .
```

## Notes

- Use `NVCCFLAGS` in the `Makefile` to tune compilation flags for your hardware.
- Mode 0: basic kernel, Mode 1: grid-stride, Mode 2: Unified Memory, Mode 3: UM with prefetch
