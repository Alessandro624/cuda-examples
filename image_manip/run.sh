#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Runner for image manipulation demos
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
if [ ! -x "./$BIN" ]; then
  echo "Building..."
  make
fi

./"$BIN" "$@"
