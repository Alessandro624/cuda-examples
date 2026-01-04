#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BIN="heat_transfer_cuda"
if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  cat <<EOF
Usage: $0 [options passed to binary]

This script runs the heat transfer CUDA simulation under nvprof and saves profiler output.

Options (passed to heat_transfer_cuda):
  --mode M         Kernel mode: 0=Global, 1=Tiled, 2=Tiled_wH
  --block BX BY    Block dimensions (8x8, 16x16, 32x32, etc.)
  --steps N        Number of simulation steps
  --verify         Verify result against CPU reference

Examples:
  ./profile_nvprof.sh --mode 0 --block 16 16
  ./profile_nvprof.sh --mode 1 --block 32 8 --steps 1000
  ./profile_nvprof.sh --mode 2 --block 16 32 --verify
EOF
  exit 0
fi

if [ ! -x "$BIN" ]; then
  echo "Building..."
  make cuda
fi

OUTFILE="nvprof_${BIN}_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "Profiling $BIN $@ -> $OUTFILE"
nvprof --print-gpu-trace --log-file "$OUTFILE" "./$BIN" "$@"
echo "Profile saved to $OUTFILE"
