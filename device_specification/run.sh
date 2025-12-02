#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Runner for device specification demo
if [ "$#" -gt 0 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  cat <<EOF
Usage: $0 [--device IDX]

If no index is provided, information for all devices is printed.

Examples:
  - $0 --device 0
  - $0 --device 1
EOF
  exit 0
fi

BIN=./deviceSpec
if [ ! -x "$BIN" ]; then
  echo "Building..."
  make
fi

"$BIN" "$@"
