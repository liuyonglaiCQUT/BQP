#!/usr/bin/env bash
source "$(dirname "$0")/activate.sh"
export PYTHONPATH="$EXT_ROOT/repos/ULTRA:${PYTHONPATH:-}"
python "$EXT_ROOT/scripts/resume_ultra.py" "$@"
