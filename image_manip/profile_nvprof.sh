#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  cat <<EOF
Usage: $0 <binary> [args...]

Available binaries: imageBlur imageToGrayscale
Default binary: imageBlur

Example: $0 imageBlur --infile=input.png --outfile=out.png
EOF
  exit 0
fi

BIN="${1:-imageBlur}"; shift || true
if [ ! -x "$BIN" ]; then
  echo "Building..."
  make
fi

OUTFILE="nvprof_${BIN}_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "Profiling ./$BIN $@ -> $OUTFILE"
nvprof --print-gpu-trace --log-file "$OUTFILE" ./$BIN "$@"
echo "Profile saved to $OUTFILE"
