#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

BIN=./parallelHistogram

if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
    cat <<EOF
Usage: $0 [OPTIONS]

Profile parallel histogram with nvprof.

Options (passed to parallelHistogram):
  --mode MODE    Kernel: naive|privatized|aggregated|coarsened|all (default: all)
  --n N          Number of input elements (default: 1048576)
  --bins BINS    Number of histogram bins (default: 256)
  --threads T    Threads per block (default: 256)
  --coarse C     Coarsening factor (default: 4)

Examples:
  $0 --mode naive --n 10000000
  $0 --mode privatized --bins 512
  $0 --mode all --n 5000000

Output:
  Creates nvprof_parallelHistogram_TIMESTAMP.log with GPU trace
EOF
    exit 0
fi

if [ ! -x "$BIN" ]; then
    echo "Building parallelHistogram..."
    make
fi

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTFILE="nvprof_parallelHistogram_${TIMESTAMP}.log"
nvprof --print-gpu-trace --log-file "$OUTFILE" "$BIN" "$@"
echo "Profile saved to: $OUTFILE"
