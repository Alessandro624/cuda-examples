#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Runner for matrixVectMul demo
BIN=./matrixVectMul
if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  cat <<EOF
Usage: $0 [--width W] [--height H] [--threads T]

Defaults: width=1024 height=1024 threads=256
EOF
  exit 0
fi

if [ ! -x "$BIN" ]; then
  echo "Building..."
  make
fi

"$BIN" "$@"
