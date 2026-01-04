#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Default to CUDA version
BIN="${BIN:-heat_transfer_cuda}"

if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  cat <<EOF
Usage: $0 [options]

This script builds and runs the heat transfer simulation.

Environment variables:
  BIN=heat_transfer_cli   Run CPU version
  BIN=heat_transfer_cuda  Run CUDA version (default)

CUDA version options:
  --mode M         Kernel mode: 0=Global, 1=Tiled, 2=Tiled_wH (default: 0)
  --block BX BY    Block dimensions (default: 16 16)
                   Supported: 8x8, 8x16, 8x32, 16x8, 16x16, 16x32, 32x8, 32x16, 32x32
  --steps N        Number of simulation steps (default: 10000)
  --rows R         Grid rows (default: 256)
  --cols C         Grid columns (default: 4096)
  --hotrows T B    Hot rows at top and bottom (default: 2 2)
  --temp T         Initial hot temperature (default: 20.0)
  --save           Save initial and final configurations
  --verify         Verify result against CPU reference

Examples:
  ./run.sh --mode 0 --block 16 16              # Global with 16x16 blocks
  ./run.sh --mode 1 --block 32 8               # Tiled with 32x8 blocks
  ./run.sh --mode 2 --block 16 32 --verify     # Tiled_wH with verification
  BIN=heat_transfer_cli ./run.sh               # Run CPU version

Visualization (gnuplot):
  plot 'temperature_step_N.dat' matrix with image

EOF
  exit 0
fi

if [ ! -x "$BIN" ]; then
  echo "Building..."
  make
fi

"./$BIN" "$@"
