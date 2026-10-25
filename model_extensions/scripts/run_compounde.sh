#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/activate.sh"
TASK="${1:-WN18RR}"
SEED="${2:-7}"
mkdir -p "$EXT_ROOT/logs"
python "$EXT_ROOT/scripts/run_compounde.py" \
  --dataset "$TASK" --seed "$SEED" "${@:3}" 2>&1 | \
  tee "$EXT_ROOT/logs/compounde_${TASK}_${SEED}_$(date +%Y%m%d_%H%M%S).log"
