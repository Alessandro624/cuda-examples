#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  cat <<EOF
Usage: $0 <binary> [args...]

Available binaries: errorCudaMemcpy vectAdd_errors
Default binary: errorCudaMemcpy

Examples: 
  - $0 vectAdd_errors --mode=2
  - $0 errorCudaMemcpy --n=1000000
  - $0 vectAdd_errors --n=500000
EOF
  exit 0
fi

BIN=${1-errorCudaMemcpy}; shift || true
if [ ! -x ./"$BIN" ]; then
  echo "Building..."
  make
fi

OUTFILE="nvprof_${BIN}_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "Profiling ./$BIN $@ -> $OUTFILE"
nvprof --print-gpu-trace --log-file "$OUTFILE" ./$BIN "$@"
echo "Profile saved to $OUTFILE"
