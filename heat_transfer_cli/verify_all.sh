#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

STEPS=10000
CPU_REF_INIT="cpu_reference_step_0.dat"
CPU_REF_FINAL="cpu_reference_step_${STEPS}.dat"

if [ "${1:-}" = "--clean" ] || [ "${1:-}" = "-c" ]; then
    echo "Removing cached CPU reference files..."
    rm -f "$CPU_REF_INIT" "$CPU_REF_FINAL"
    shift
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<EOF
Usage: $0 [options]

Verifies all CUDA implementations against CPU reference.

Options:
  --clean, -c    Force regeneration of CPU reference files
  --help, -h     Show this help message

The CPU reference files are cached after first run to speed up subsequent verifications.
EOF
    exit 0
fi

echo "=============================================="
echo "Heat Transfer Verification Script"
echo "=============================================="
echo "Steps: $STEPS"
echo "=============================================="

echo ""
echo "[BUILD] Building CPU and CUDA versions..."
make all

if [ -f "$CPU_REF_INIT" ] && [ -f "$CPU_REF_FINAL" ]; then
    echo ""
    echo "[CPU] Using cached CPU reference files..."
    CPU_INIT_CHECKSUM=$(md5sum "$CPU_REF_INIT" | awk '{print $1}')
    CPU_FINAL_CHECKSUM=$(md5sum "$CPU_REF_FINAL" | awk '{print $1}')
else
    echo ""
    echo "[CPU] Running CPU reference version (first time, will be cached)..."
    rm -f temperature_step_*.dat 2>/dev/null || true
    ./heat_transfer_cli

    if [ ! -f temperature_step_0.dat ] || [ ! -f temperature_step_${STEPS}.dat ]; then
        echo "[ERROR] CPU output files not found!"
        exit 1
    fi

    mv temperature_step_0.dat "$CPU_REF_INIT"
    mv temperature_step_${STEPS}.dat "$CPU_REF_FINAL"
    
    CPU_INIT_CHECKSUM=$(md5sum "$CPU_REF_INIT" | awk '{print $1}')
    CPU_FINAL_CHECKSUM=$(md5sum "$CPU_REF_FINAL" | awk '{print $1}')
fi

echo "[CPU] Initial config checksum:  $CPU_INIT_CHECKSUM"
echo "[CPU] Final config checksum:    $CPU_FINAL_CHECKSUM"
echo "[CPU] Reference files: $CPU_REF_INIT, $CPU_REF_FINAL"

MODES=(0 1 2)
MODE_NAMES=("Global" "Tiled" "Tiled_wH")
BLOCK_CONFIGS=("8 8" "8 16" "8 32" "16 8" "16 16" "16 32" "32 8" "32 16" "32 32")

PASSED=0
FAILED=0
TOTAL=0

echo ""
echo "=============================================="
echo "[CUDA] Testing all configurations..."
echo "=============================================="

for mode_idx in "${!MODES[@]}"; do
    mode=${MODES[$mode_idx]}
    mode_name=${MODE_NAMES[$mode_idx]}
    
    for block_config in "${BLOCK_CONFIGS[@]}"; do
        bx=$(echo $block_config | awk '{print $1}')
        by=$(echo $block_config | awk '{print $2}')
        
        TOTAL=$((TOTAL + 1))
        config_name="Mode=$mode_name Block=${bx}x${by}"
        
        ./heat_transfer_cuda --mode $mode --block $bx $by --save >/dev/null 2>&1
        
        CUDA_INIT_CHECKSUM=$(md5sum temperature_step_0.dat 2>/dev/null | awk '{print $1}' || echo "MISSING")
        CUDA_FINAL_CHECKSUM=$(md5sum temperature_step_${STEPS}.dat 2>/dev/null | awk '{print $1}' || echo "MISSING")
        
        if [ "$CUDA_INIT_CHECKSUM" = "$CPU_INIT_CHECKSUM" ] && [ "$CUDA_FINAL_CHECKSUM" = "$CPU_FINAL_CHECKSUM" ]; then
            echo "[PASS] $config_name"
            PASSED=$((PASSED + 1))
        else
            echo "[FAIL] $config_name"
            echo "       Init:  CPU=$CPU_INIT_CHECKSUM CUDA=$CUDA_INIT_CHECKSUM"
            echo "       Final: CPU=$CPU_FINAL_CHECKSUM CUDA=$CUDA_FINAL_CHECKSUM"
            FAILED=$((FAILED + 1))
        fi
        
        rm -f temperature_step_0.dat temperature_step_${STEPS}.dat
    done
done

echo ""
echo "=============================================="
echo "VERIFICATION SUMMARY"
echo "=============================================="
echo "Total:  $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "=============================================="
echo "Note: CPU reference files are cached."
echo "      Delete $CPU_REF_INIT and $CPU_REF_FINAL to regenerate."
echo "=============================================="

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "✓ All CUDA implementations produce identical results to CPU!"
    exit 0
else
    echo ""
    echo "✗ Some CUDA implementations produced different results!"
    exit 1
fi
