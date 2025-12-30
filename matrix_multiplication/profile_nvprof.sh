#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

BIN=./matrixMul

if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
    cat <<EOF
Usage: $0 [OPTIONS]

Profile matrix multiplication with nvprof.

Options (passed to matrixMul):
  --mode MODE       Kernel mode: naive|tiled|coarsened|perrows|percols (default: tiled)
  --M M             Number of rows in A and C (default: 2048)
  --K K             Number of cols in A / rows in B (default: 1024)
  --N N             Number of cols in B and C (default: 512)
  --threads T       Threads per block (default: 256)
  --tile TILE       Tile width for tiled kernels (default: 16)
  --coarse-factor F Coarsening factor (default: 4)

Examples:
  $0 --mode naive --M 1024 --K 1024 --N 1024
  $0 --mode tiled --tile 32 --M 2048 --K 2048 --N 2048
  $0 --mode coarsened --tile 16 --coarse-factor 4

Output:
  Creates nvprof_matrixMul_TIMESTAMP.log with GPU trace
EOF
    exit 0
fi

if [ ! -x "$BIN" ]; then
    echo "Building matrixMul..."
    make
fi

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTFILE="nvprof_matrixMul_${TIMESTAMP}.log"
nvprof --print-gpu-trace --log-file "$OUTFILE" "$BIN" "$@"
echo "Profile saved to: $OUTFILE"
