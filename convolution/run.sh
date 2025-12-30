#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Runner for convolution examples

show_help() {
    cat <<EOF
Usage: $0 <1D|2D> [OPTIONS]

Run convolution examples.

Commands:
  1D    Run 1D convolution
  2D    Run 2D convolution

Options for 1D:
  --n N          Length of signal (default: 1000000)

Options for 2D:
  --width W      Image width (default: 1920)
  --height H     Image height (default: 1080)

Examples:
  $0 1D --n 2000000
  $0 2D --width 3840 --height 2160
EOF
    exit 0
}

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
fi

MODE="$1"
shift

case "$MODE" in
    1D|1d)
        BIN=./convolution1D
        if [ ! -x "$BIN" ]; then
            echo "Building convolution1D..."
            make convolution1D
        fi
        "$BIN" "$@"
        ;;
    2D|2d)
        BIN=./convolution2D
        if [ ! -x "$BIN" ]; then
            echo "Building convolution2D..."
            make convolution2D
        fi
        "$BIN" "$@"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'. Use '1D' or '2D'."
        exit 1
        ;;
esac
