#!/usr/bin/env bash
source "$(dirname "$0")/activate.sh"
TASK="${1:-WN18RR}"; MODE="${2:-eval}"; SEED="${3:-7}"
[[ "$TASK" == WN18RR || "$TASK" == FB15k237 ]] || exit 2
[[ "$MODE" == eval || "$MODE" == finetune ]] || exit 2
OUT="$EXT_ROOT/runs/ULTRA/$TASK/${MODE}_seed_$SEED"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT"; exit 2; }
mkdir -p "$OUT"
python "$EXT_ROOT/scripts/make_ultra_config.py" --dataset "$TASK" --mode "$MODE" --out "$OUT/config.yaml"
export PYTHONPATH="$EXT_ROOT/repos/ULTRA:${PYTHONPATH:-}"
cd "$OUT"
python "$EXT_ROOT/repos/ULTRA/script/run.py" -c "$OUT/config.yaml" --seed "$SEED" 2>&1 | tee "$OUT/console.log"
