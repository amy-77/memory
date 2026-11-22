#!/usr/bin/env bash
# Run/resume to the configured total. No external checkout, hidden retries or checkpoint deletion.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_8gpu.sh" "$@"
