#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

BIN=./stencil

if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
    cat <<EOF
Usage: $0 [OPTIONS]

Profile 3D seven-point stencil with nvprof.

Options (passed to stencil):
  --mode MODE    Kernel: naive|shared|coarsened|register|all (default: all)
  --nx NX        Grid size in X (default: 256)
  --ny NY        Grid size in Y (default: 256)
  --nz NZ        Grid size in Z (default: 256)

Examples:
  $0 --mode naive --nx 256 --ny 256 --nz 256
  $0 --mode shared --nx 512 --ny 512 --nz 256
  $0 --mode register
  $0 --mode all

Output:
  Creates nvprof_stencil_TIMESTAMP.log with GPU trace
EOF
    exit 0
fi

if [ ! -x "$BIN" ]; then
    echo "Building stencil..."
    make
fi

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTFILE="nvprof_stencil_${TIMESTAMP}.log"
nvprof --print-gpu-trace --log-file "$OUTFILE" "$BIN" "$@"
echo "Profile saved to: $OUTFILE"
