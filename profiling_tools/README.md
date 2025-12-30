# CUDA Profiling Tools with Roofline Analysis

Portable profiling suite for CUDA applications with roofline model visualization.

## Features

- **Automated profiling** of multiple CUDA executables
- **GPU specifications** detection and theoretical peak calculation
- **Roofline model** visualization (DRAM, L2, Shared memory)
- **Performance metrics**: execution time, occupancy, arithmetic intensity
- **Flexible filtering** by executables and kernels
- **Multi-format output**: CSV data, plots (PNG/SVG), and data files

## Files

| File | Description |
|------|-------------|
| `profile_cuda.sh` | Main profiling orchestration script |
| `gpu_info.cu` | GPU specification detection tool (CUDA source) |
| `parse_metrics.py` | Python script for parsing profiler output and generating data files |
| `plot_roofline.gp` | Gnuplot script for roofline model visualization |
| `plot_histogram.gp` | Gnuplot script for execution time histograms |
| `plot_occupancy.gp` | Gnuplot script for SM occupancy bar charts |
| `Makefile` | Build rules for `gpu_info` |
| `README.md` | This documentation file |

## Requirements

### Essential

- **CUDA Toolkit** (with `nvcc`)
- **Python 3.6+**
- **NVIDIA GPU** with compute capability 3.0+

### Profiling Tools (one of)

- **Nsight Compute** (`ncu`) - Recommended for modern GPUs
- **nvprof** - Legacy tool (deprecated but still functional)

### Optional

- **gnuplot** - For automated plot generation

## Installation

1. Clone or copy the profiling tools to your repository:

```bash
mkdir profiling_tools
cd profiling_tools
# Copy all files here
```

2. Make scripts executable:

```bash
chmod +x profile_cuda.sh
```

3. Compile the GPU info tool:

```bash
nvcc -o gpu_info gpu_info.cu
```

## Usage

### Basic Usage

Profile all executables in a directory:

```bash
./profile_cuda.sh -d /path/to/executables
# or
./profile_cuda.sh --dir /path/to/executables
```

### With Arguments

Pass arguments to your executables:

```bash
./profile_cuda.sh -d ./build -a "--n 1024 --threads 256"
# or
./profile_cuda.sh --dir ./build --args "--n 1024 --threads 256"
```

### Filter by Executables

Profile only specific executables:

```bash
./profile_cuda.sh -d ./build -e "app1,app2,app3"
# or
./profile_cuda.sh --dir ./build --executables "app1,app2,app3"
```

### Filter by Kernels

Profile only specific kernels:

```bash
./profile_cuda.sh -d ./build -k "matmul,reduction,scan"
# or
./profile_cuda.sh --dir ./build --kernels "matmul,reduction,scan"
```

### Custom Output Directory

Specify output directory name:

```bash
./profile_cuda.sh -d ./build -o "experiment1"
# or
./profile_cuda.sh --dir ./build --output "experiment1"
```

### Complete Example

```bash
./profile_cuda.sh \
    -d ./cuda_apps \
    -e "solver_v1,solver_v2,solver_optimized" \
    -k "computeFlux,updateCells" \
    -a "--config config.ini --iterations 10000" \
    -o "performance_study"
```

### Help

Show all available options:

```bash
./profile_cuda.sh -h
# or
./profile_cuda.sh --help
```

## Output Structure

Profiling creates a timestamped directory with all results:

```
your_exec_dir/
└── profiling_results_20240101_120000/
    ├── gpu_info.log                    # GPU specifications
    ├── roofline_specs.gp              # Specs for gnuplot
    ├── executable1_summary.csv         # Timing summary
    ├── executable1_compute.csv         # FLOP counts
    ├── executable1_memory.csv          # Memory transactions
    ├── executable1_occupancy.csv       # Occupancy metrics
    ├── executable1.log                 # Execution output
    ├── roofline_data.dat              # Roofline plot data
    ├── time_data.dat                  # Timing bar chart data
    ├── occupancy_data.dat             # Occupancy data
    ├── roofline_fp64.png              # Roofline plot
    ├── histogram_times.png            # Timing comparison
    └── occupancy.png                  # Occupancy comparison
```

## Understanding the Outputs

### Roofline Plot

The roofline plot shows:

- **X-axis**: Arithmetic Intensity (FLOP/Byte)
- **Y-axis**: Performance (GFLOP/s)
- **Roofline lines**: Memory bandwidth bounds (DRAM, L2, Shared)
- **Horizontal line**: Peak compute performance
- **Points**: Your kernels' actual performance

**Interpretation**:

- Points below DRAM line: Memory bound
- Points below peak line but above memory lines: Compute bound
- Closer to rooflines = better performance

### Timing Histogram

Bar chart comparing total execution time across versions.

### Occupancy Plot

Shows SM occupancy (0-1) for each version:

- **1.0**: Perfect occupancy
- **0.5**: Half of available warps active
- **< 0.3**: Poor occupancy, may limit performance

## GPU Information Tool

Run standalone to get GPU specs:

```bash
./gpu_info
```

Output includes:

- Architecture (Ampere, Turing, Pascal, etc.)
- CUDA cores and clock speeds
- Memory hierarchy specifications
- Theoretical peak bandwidths and FLOP/s

Select specific GPU:

```bash
./gpu_info 1  # Use GPU device 1
```

## Customization

### Modify Default Metrics

Edit `profile_cuda.sh` to add/remove nvprof/ncu metrics:

```bash
# Add more compute metrics
ncu --metrics sm__inst_executed.sum,sm__cycles_active.sum ...

# Add more memory metrics
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum ...
```

### Change Plot Styles

Edit `plot_*.gp` files to customize:

- Colors and line styles
- Axis ranges and labels
- Output formats (PNG, SVG, PDF)

### Adjust Parsing Logic

Edit `parse_metrics.py` to:

- Change kernel name extraction logic
- Modify metric aggregation
- Add custom calculations

## Common Issues

### "No executables found"

- Ensure your directory contains executable files
- Check file permissions (`chmod +x your_app`)
- Use `--executables` to specify names explicitly

### "Neither ncu nor nvprof found"

- Install CUDA Toolkit completely
- Add CUDA bin directory to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

### Profiling takes too long

- Reduce number of iterations in your application
- Use `--kernels` to profile only specific kernels
- Profile with smaller input datasets

### No plots generated

- Install gnuplot: `sudo apt install gnuplot` (Ubuntu) or `brew install gnuplot` (macOS)
- Check that `.dat` files were created
- Run gnuplot manually: `gnuplot plot_roofline.gp`

### "Permission denied" errors

- Make scripts executable: `chmod +x *.sh`
- Check write permissions in target directory

## Performance Tips

1. **Reduce profiling overhead**: Use reduced iteration counts for profiling runs
2. **Focus on hotspots**: Profile only performance-critical kernels
3. **Compare versions**: Keep consistent input sizes when comparing implementations
4. **Check occupancy**: Low occupancy often indicates optimization opportunities
5. **Memory patterns**: High AI suggests compute-bound, low AI suggests memory-bound

## Advanced Usage

### Batch Processing

Profile multiple configurations:

```bash
for config in config1.ini config2.ini config3.ini; do
    ./profile_cuda.sh --dir ./build --args "$config output 1000" --output "run_$config"
done
```

### Custom Metrics

Add application-specific metrics to `parse_metrics.py`:

```python
def calculate_custom_metric(kernels):
    # Example: Calculate bandwidth utilization
    actual_bw = total_bytes / total_time / 1e9
    utilization = actual_bw / specs['bw_dram']
    return utilization
```
