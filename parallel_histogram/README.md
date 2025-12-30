# Parallel Histogram

CUDA implementations of parallel histogram computation demonstrating various optimization strategies for handling output conflicts with atomic operations.

## Build

```bash
cd parallel_histogram
make
```

## Usage

```bash
./parallelHistogram [--mode MODE] [--n N] [--bins BINS] [--threads T] [--coarse C]
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Kernel mode: `naive`, `privatized`, `aggregated`, `coarsened`, `all` | `all` |
| `--n` | Number of input elements | 1048576 (1M) |
| `--bins` | Number of histogram bins | 256 |
| `--threads` | Threads per block | 256 |
| `--coarse` | Coarsening factor (elements per thread) | 4 |

## Kernel Implementations

### 1. Naive (`histogram_naive`)

Basic implementation where each thread processes one element and performs an atomic add directly to global memory.

```
Thread i → atomicAdd(&global_histogram[input[i]], 1)
```

**Characteristics:**
- Simple implementation
- High atomic contention on global memory
- Performance limited by atomic serialization

### 2. Privatized (`histogram_privatized`)

Each block maintains a private histogram in shared memory. Atomics operate on shared memory (faster), then results are merged to global memory.

```
Phase 1: atomicAdd(&shared_histogram[bin], 1)  // Low contention
Phase 2: atomicAdd(&global_histogram[bin], shared_histogram[bin])  // One per block
```

**Characteristics:**

- Reduces global memory atomic contention by factor of blocks
- Requires shared memory proportional to number of bins
- Best for moderate number of bins (≤ 4096)

### 3. Aggregated (`histogram_aggregated`)

Before performing atomics, threads aggregate counts for consecutive identical values. Useful when input has local patterns.

```
if (current_bin == previous_bin):
    count++
else:
    atomicAdd(&histogram[previous_bin], count)
    count = 1
```

**Characteristics:**
- Reduces atomic operations when input has locality
- Combines with privatization for best results
- Overhead may not pay off for random data

### 4. Coarsened (`histogram_coarsened`)

Each thread processes multiple consecutive elements (coarsening factor). Thread maintains local accumulators before committing to shared memory.

```
Thread i processes: input[i*C], input[i*C+1], ..., input[i*C+(C-1)]
```

**Characteristics:**
- Reduces total number of atomic operations
- Better memory access patterns (coalescing)
- Local aggregation within each thread's elements

## Examples

```bash
# Run all kernels with default parameters
./parallelHistogram

# Run only privatized kernel with larger input
./parallelHistogram --mode privatized --n 10000000

# Run coarsened kernel with custom parameters
./parallelHistogram --mode coarsened --n 5000000 --coarse 8 --threads 128

# Use fewer bins
./parallelHistogram --bins 64 --n 1000000
```

## Run

```bash
./run.sh [OPTIONS]
```

## Profiling

```bash
# Profile with nvprof
./profile_nvprof.sh --mode all --n 10000000

# Use profiling tools
../profiling_tools/profile_cuda.sh -d .
```

## Notes

- Maximum bins limited to 4096 due to shared memory constraints.
- Privatized kernel provides best performance for typical use cases.
- Coarsened kernel benefits from better memory coalescing.
- Host-side verification ensures correctness.
