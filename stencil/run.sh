#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Runner for 3D Seven-Point Stencil example

BIN=./stencil

if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
    cat <<EOF
Usage: $0 [OPTIONS]

Run 3D seven-point stencil example.

Options (passed to stencil):
  --mode MODE    Kernel: naive|shared|coarsened|register|all (default: all)
  --nx NX        Grid size in X (default: 256)
  --ny NY        Grid size in Y (default: 256)
  --nz NZ        Grid size in Z (default: 256)

Examples:
  $0 --mode naive --nx 128 --ny 128 --nz 128
  $0 --mode shared --nx 512 --ny 512 --nz 256
  $0 --mode register
  $0 --mode all --nx 256 --ny 256 --nz 256
EOF
    exit 0
fi

if [ ! -x "$BIN" ]; then
    echo "Building stencil..."
    make
fi

"$BIN" "$@"
