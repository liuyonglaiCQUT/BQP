#!/usr/bin/env bash
source "$(dirname "$0")/activate.sh"
TASK="${1:-WN18RR}"; SEED="${2:-7}"
[[ "$TASK" == WN18RR || "$TASK" == FB15k237 || "$TASK" == NELL995 ]] || exit 2
export OUTPUT_DIR="$EXT_ROOT/runs/SimKGC/$TASK/seed_$SEED"
export DATA_DIR="$EXT_ROOT/repos/SimKGC/data/$TASK"
CHECKPOINT="$OUTPUT_DIR/model_last.mdl"
[[ -f "$CHECKPOINT" ]] || { echo "No resumable checkpoint: $CHECKPOINT"; exit 2; }
export SIMKGC_GRAD_CHECKPOINT=1
PRETRAINED_MODEL="${SIMKGC_PRETRAINED_MODEL:-bert-base-uncased}"
if [[ -z "${SIMKGC_PRETRAINED_MODEL:-}" && -f "$EXT_ROOT/bert-base-uncased/config.json" ]]; then
  PRETRAINED_MODEL="$EXT_ROOT/bert-base-uncased"
fi
exec > >(tee -a "$OUTPUT_DIR/console.log") 2>&1
cd "$EXT_ROOT/repos/SimKGC"
python "$EXT_ROOT/scripts/prepare_simkgc.py" "$TASK"
SCRIPT=train_wn.sh; [[ "$TASK" == FB15k237 ]] && SCRIPT=train_fb.sh
if [[ "$TASK" == NELL995 ]]; then
  bash "$EXT_ROOT/scripts/train_simkgc_nell995.sh" --seed "$SEED" \
    --batch-size 1024 --max-to-keep 1 --pretrained-model "$PRETRAINED_MODEL" \
    --resume "$CHECKPOINT"
else
  bash "scripts/$SCRIPT" --seed "$SEED" --batch-size 1024 --max-to-keep 1 \
    --pretrained-model "$PRETRAINED_MODEL" --resume "$CHECKPOINT"
fi
bash scripts/eval.sh "$OUTPUT_DIR/model_last.mdl" "$TASK" "$DATA_DIR/valid.txt.json"
bash scripts/eval.sh "$OUTPUT_DIR/model_last.mdl" "$TASK"
