#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Runner for matrixMul demo
BIN=./matrixMul
if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  cat <<EOF
Usage: $0 [--mode MODE] [--M M] [--K K] [--N N] [--threads THREADS] [--tile TILE] [--coarse-factor COARSE]

All arguments are optional. Flags may be provided in any order.

Modes: naive tiled shared coarsened perrows percols
Defaults: mode=tiled, M=2048, K=1024, N=512, THREADS=256, TILE=16, COARSE=4

Examples:
  $0                         # run tiled 2048x1024x512 (defaults)
  $0 --mode=naive --M=512    # naive with M=512 (K,N default)
EOF
  exit 0
fi

if [ ! -x "$BIN" ]; then
  echo "Building..."
  make
fi

"$BIN" "$@"
