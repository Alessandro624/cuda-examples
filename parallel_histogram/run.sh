#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Runner for parallel histogram example

BIN=./parallelHistogram

if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
    cat <<EOF
Usage: $0 [OPTIONS]

Run parallel histogram example.

Options (passed to parallelHistogram):
  --mode MODE    Kernel: naive|privatized|aggregated|coarsened|all (default: all)
  --n N          Number of input elements (default: 1048576)
  --bins BINS    Number of histogram bins (default: 256)
  --threads T    Threads per block (default: 256)
  --coarse C     Coarsening factor (default: 4)

Examples:
  $0 --mode privatized --n 10000000
  $0 --mode coarsened --coarse 8
  $0 --mode all --bins 512
EOF
    exit 0
fi

if [ ! -x "$BIN" ]; then
    echo "Building parallelHistogram..."
    make
fi

"$BIN" "$@"
