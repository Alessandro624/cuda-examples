#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

BIN=./vectAdd
if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  cat <<EOF
Usage: $0 [--mode M] [--n N] [--threads T] [--granularity G]

Modes: 0=host copy, 1=grid-stride, 2=managed, 3=managed+prefetch
Defaults: mode=0 n=1048576 threads=1024 granularity=1
EOF
  exit 0
fi

if [ ! -x "$BIN" ]; then
  echo "Building..."
  make
fi

OUTFILE="nvprof_vectAdd_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "Profiling $BIN "$@" -> $OUTFILE"
nvprof --print-gpu-trace --log-file "$OUTFILE" "$BIN" "$@"
echo "Profile saved to $OUTFILE"
