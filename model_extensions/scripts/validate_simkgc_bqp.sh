#!/usr/bin/env bash
source "$(dirname "$0")/activate.sh"
python "$EXT_ROOT/scripts/validate_simkgc_bqp.py" "$@"
