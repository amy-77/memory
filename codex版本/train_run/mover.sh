#!/usr/bin/env bash
# Copy by default. Pass --move to remove verified local large files; optimizer history is kept.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${RUN_DIR:?Set RUN_DIR to the training output directory}"
: "${ARCHIVE_DIR:?Set ARCHIVE_DIR to a POSIX-mounted checkpoint archive}"
exec "${PYTHON:-python}" "$SCRIPT_DIR/../framework/scripts/archive_checkpoints.py" \
  "$RUN_DIR/ckpt" "$ARCHIVE_DIR" "$@"
