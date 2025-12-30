#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

show_help() {
    cat <<EOF
Usage: $0 <1D|2D> [OPTIONS]

Profile convolution kernels with nvprof.

Commands:
  1D    Profile 1D convolution kernels
  2D    Profile 2D convolution kernels

Options for 1D:
  --n N          Length of signal (default: 1000000)

Options for 2D:
  --width W      Image width (default: 1920)
  --height H     Image height (default: 1080)

Examples:
  $0 1D --n 2000000
  $0 2D --width 1920 --height 1080

Output:
  Creates nvprof_convolution{1D|2D}_TIMESTAMP.log with GPU trace
EOF
    exit 0
}

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
fi

MODE="$1"
shift

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

case "$MODE" in
    1D|1d)
        BIN=./convolution1D
        if [ ! -x "$BIN" ]; then
            echo "Building convolution1D..."
            make convolution1D
        fi
        OUTFILE="nvprof_convolution1D_${TIMESTAMP}.log"
        nvprof --print-gpu-trace --log-file "$OUTFILE" "$BIN" "$@"
        ;;
    2D|2d)
        BIN=./convolution2D
        if [ ! -x "$BIN" ]; then
            echo "Building convolution2D..."
            make convolution2D
        fi
        OUTFILE="nvprof_convolution2D_${TIMESTAMP}.log"
        nvprof --print-gpu-trace --log-file "$OUTFILE" "$BIN" "$@"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'. Use '1D' or '2D'."
        exit 1
        ;;
esac

echo "Profile saved to: $OUTFILE"
