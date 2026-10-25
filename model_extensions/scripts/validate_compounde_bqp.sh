#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/activate.sh"
python "$EXT_ROOT/scripts/validate_compounde_bqp.py" "$@"
