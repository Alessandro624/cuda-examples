#!/bin/bash

# =============================================================================
# CUDA Profiling Tool with Roofline Analysis
# =============================================================================
# Usage: ./profile_cuda.sh [OPTIONS]
# Options:
#   -d, --dir <path>          Directory containing executables (required)
#   -e, --executables <list>  Comma-separated list of executables (default: all)
#   -k, --kernels <list>      Comma-separated list of kernel names to profile (default: all)
#   -a, --args <"args">       Arguments to pass to executables (in quotes)
#   -o, --output <name>       Output directory name (default: profiling_results_TIMESTAMP)
#   -p, --precision <fp32|fp64> Precision for roofline analysis (default: fp64)
#   -h, --help                Show this help message
#
# Example:
#   ./profile_cuda.sh -d ./build -e "app1,app2" -k "kernel1,kernel2" -a "input.cfg output 1000"
# =============================================================================

set -e

# Default values
EXEC_DIR=""
EXECUTABLES=""
KERNELS=""
EXEC_ARGS=""
OUTPUT_NAME=""
PRECISION="fp64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            EXEC_DIR="$2"
            shift 2
            ;;
        -e|--executables)
            EXECUTABLES="$2"
            shift 2
            ;;
        -k|--kernels)
            KERNELS="$2"
            shift 2
            ;;
        -a|--args)
            EXEC_ARGS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        -p|--precision)
            PRECISION="$2"
            if [[ "$PRECISION" != "fp32" && "$PRECISION" != "fp64" ]]; then
                echo -e "${RED}Error: Precision must be 'fp32' or 'fp64'${NC}"
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            grep "^# " "$0" | cut -c 3-
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$EXEC_DIR" ]; then
    echo -e "${RED}Error: Executable directory (-d/--dir) is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -d "$EXEC_DIR" ]; then
    echo -e "${RED}Error: Directory '$EXEC_DIR' does not exist${NC}"
    exit 1
fi

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$OUTPUT_NAME" ]; then
    OUTPUT_DIR="${EXEC_DIR}/profiling_results_${TIMESTAMP}"
else
    OUTPUT_DIR="${EXEC_DIR}/${OUTPUT_NAME}_${TIMESTAMP}"
fi

mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}Created output directory: $OUTPUT_DIR${NC}"

# Find executables
if [ -z "$EXECUTABLES" ]; then
    # Find all executables in directory (excluding hidden files and scripts)
    EXEC_LIST=$(find "$EXEC_DIR" -maxdepth 1 -type f -executable ! -name ".*" ! -name "*.sh" ! -name "*.py" -printf "%f\n")
    if [ -z "$EXEC_LIST" ]; then
        echo -e "${RED}Error: No executables found in $EXEC_DIR${NC}"
        exit 1
    fi
else
    # Convert comma-separated list to newline-separated
    EXEC_LIST=$(echo "$EXECUTABLES" | tr ',' '\n')
fi

echo -e "${BLUE}Executables to profile:${NC}"
echo "$EXEC_LIST" | while read exe; do echo "  - $exe"; done

# Check for nvprof/ncu
PROFILER=""
if command -v nvprof &> /dev/null; then
    PROFILER="nvprof"
    echo -e "${YELLOW}Using legacy nvprof (consider upgrading to Nsight Compute)${NC}"
elif command -v ncu &> /dev/null; then
    PROFILER="ncu"
    echo -e "${GREEN}Using NVIDIA Nsight Compute (ncu)${NC}"
else
    echo -e "${RED}Error: Neither ncu nor nvprof found. Please install CUDA profiling tools.${NC}"
    exit 1
fi

# =============================================================================
# Step 1: Get GPU Information
# =============================================================================
echo ""
echo "=========================================================="
echo " [1/3] COLLECTING GPU INFORMATION"
echo "=========================================================="

# Try to compile and run gpumembench if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/gpu_info.cu" ]; then
    echo "Compiling GPU info tool..."
    nvcc -o "$OUTPUT_DIR/gpu_info" "$SCRIPT_DIR/gpu_info.cu" 2>/dev/null || \
        echo -e "${YELLOW}Warning: Could not compile gpu_info, will use nvidia-smi${NC}"
fi

if [ -f "$OUTPUT_DIR/gpu_info" ]; then
    "$OUTPUT_DIR/gpu_info" > "$OUTPUT_DIR/gpu_info.log"
    echo -e "${GREEN}GPU specifications collected${NC}"
else
    # Fallback to nvidia-smi
    nvidia-smi --query-gpu=name,memory.total,memory.bus_width,clocks.mem --format=csv > "$OUTPUT_DIR/gpu_info.log" 2>&1 || \
        echo -e "${YELLOW}Warning: Could not get GPU info${NC}"
fi

# =============================================================================
# Step 2: Profile Each Executable
# =============================================================================
echo ""
echo "=========================================================="
echo " [2/3] PROFILING EXECUTABLES"
echo "=========================================================="

# Build kernel filter for profiler
KERNEL_FILTER=""
if [ -n "$KERNELS" ]; then
    if [ "$PROFILER" = "ncu" ]; then
        # For ncu: --kernel-name regex
        KERNEL_PATTERN=$(echo "$KERNELS" | tr ',' '|')
        KERNEL_FILTER="--kernel-name-base regex:($KERNEL_PATTERN)"
    else
        # For nvprof: --kernels
        KERNEL_FILTER="--kernels $(echo $KERNELS | tr ',' ':')"
    fi
    echo -e "${BLUE}Filtering kernels: $KERNELS${NC}"
fi

for exe_name in $EXEC_LIST; do
    exe_path="$EXEC_DIR/$exe_name"
    
    if [ ! -f "$exe_path" ]; then
        echo -e "${YELLOW}Warning: $exe_name not found, skipping${NC}"
        continue
    fi
    
    if [ ! -x "$exe_path" ]; then
        echo -e "${YELLOW}Warning: $exe_name is not executable, skipping${NC}"
        continue
    fi
    
    echo ""
    echo "----------------------------------------------------------"
    echo " Profiling: $exe_name"
    echo "----------------------------------------------------------"
    
    BASE_NAME=$(basename "$exe_name")
    
    if [ "$PROFILER" = "ncu" ]; then
        # Use Nsight Compute
        echo "[1/4] Collecting summary metrics..."
        ncu --csv --log-file "${OUTPUT_DIR}/${BASE_NAME}_summary.csv" \
            $KERNEL_FILTER \
            --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
            "$exe_path" $EXEC_ARGS > "${OUTPUT_DIR}/${BASE_NAME}.log" 2>&1 || true
        
        echo "[2/4] Collecting compute metrics..."
        ncu --csv --log-file "${OUTPUT_DIR}/${BASE_NAME}_compute.csv" \
            $KERNEL_FILTER \
            --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum \
            "$exe_path" $EXEC_ARGS > /dev/null 2>&1 || true
        
        echo "[3/4] Collecting memory metrics..."
        ncu --csv --log-file "${OUTPUT_DIR}/${BASE_NAME}_memory.csv" \
            $KERNEL_FILTER \
            --metrics dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum \
            "$exe_path" $EXEC_ARGS > /dev/null 2>&1 || true
        
        echo "[4/4] Collecting occupancy..."
        ncu --csv --log-file "${OUTPUT_DIR}/${BASE_NAME}_occupancy.csv" \
            $KERNEL_FILTER \
            --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
            "$exe_path" $EXEC_ARGS > /dev/null 2>&1 || true
    else
        # Use legacy nvprof
        echo "[1/4] Measuring execution time..."
        nvprof --print-gpu-summary --log-file "${OUTPUT_DIR}/${BASE_NAME}_summary.csv" --csv \
            $KERNEL_FILTER \
            "$exe_path" $EXEC_ARGS > "${OUTPUT_DIR}/${BASE_NAME}.log" 2>&1 || true
        
        echo "[2/4] Collecting compute metrics..."
        nvprof --metrics flop_count_dp,flop_count_sp,flop_count_hp \
            --log-file "${OUTPUT_DIR}/${BASE_NAME}_compute.csv" --csv \
            $KERNEL_FILTER \
            "$exe_path" $EXEC_ARGS > /dev/null 2>&1 || true
        
        echo "[3/4] Collecting memory metrics..."
        nvprof --metrics gld_transactions,gst_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions \
            --log-file "${OUTPUT_DIR}/${BASE_NAME}_memory.csv" --csv \
            $KERNEL_FILTER \
            "$exe_path" $EXEC_ARGS > /dev/null 2>&1 || true
        
        echo "[4/4] Collecting occupancy..."
        nvprof --metrics achieved_occupancy \
            --log-file "${OUTPUT_DIR}/${BASE_NAME}_occupancy.csv" --csv \
            $KERNEL_FILTER \
            "$exe_path" $EXEC_ARGS > /dev/null 2>&1 || true
    fi
    
    echo -e "${GREEN}✓ Profiling complete for $exe_name${NC}"
done

# =============================================================================
# Step 3: Parse and Visualize
# =============================================================================
echo ""
echo "=========================================================="
echo " [3/3] PARSING METRICS AND GENERATING PLOTS"
echo "=========================================================="

# Copy parsing script to output directory
if [ -f "$SCRIPT_DIR/parse_metrics.py" ]; then
    cp "$SCRIPT_DIR/parse_metrics.py" "$OUTPUT_DIR/"
    cd "$OUTPUT_DIR"
    python3 parse_metrics.py --precision "$PRECISION" || echo -e "${YELLOW}Warning: Parsing failed${NC}"
    
    # Generate plots if gnuplot is available
    if command -v gnuplot &> /dev/null; then
        for gp_script in "$SCRIPT_DIR"/plot_*.gp; do
            if [ -f "$gp_script" ]; then
                cp "$gp_script" .
                gnuplot "$(basename "$gp_script")" 2>/dev/null || echo -e "${YELLOW}Warning: Plot $(basename "$gp_script") failed${NC}"
            fi
        done
        echo -e "${GREEN}✓ Plots generated${NC}"
    else
        echo -e "${YELLOW}Warning: gnuplot not found, skipping visualization${NC}"
    fi
else
    echo -e "${YELLOW}Warning: parse_metrics.py not found${NC}"
fi

echo ""
echo -e "${GREEN}=========================================================="
echo " PROFILING COMPLETE"
echo "==========================================================${NC}"
echo -e "Results saved in: ${BLUE}$OUTPUT_DIR${NC}"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.{csv,log,dat,png,svg} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
