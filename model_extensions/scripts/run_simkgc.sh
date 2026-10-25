#!/usr/bin/env bash
source "$(dirname "$0")/activate.sh"
TASK="${1:-WN18RR}"; SEED="${2:-7}"
[[ "$TASK" == WN18RR || "$TASK" == FB15k237 || "$TASK" == NELL995 ]] || exit 2
export OUTPUT_DIR="$EXT_ROOT/runs/SimKGC/$TASK/seed_$SEED"
export DATA_DIR="$EXT_ROOT/repos/SimKGC/data/$TASK"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Refusing to overwrite $OUTPUT_DIR"; exit 2; }
export SIMKGC_GRAD_CHECKPOINT=1
PRETRAINED_MODEL="${SIMKGC_PRETRAINED_MODEL:-bert-base-uncased}"
if [[ -z "${SIMKGC_PRETRAINED_MODEL:-}" && -f "$EXT_ROOT/bert-base-uncased/config.json" ]]; then
  PRETRAINED_MODEL="$EXT_ROOT/bert-base-uncased"
fi
cd "$EXT_ROOT/repos/SimKGC"
python "$EXT_ROOT/scripts/prepare_simkgc.py" "$TASK"
# Finish the resumable Hugging Face download before claiming a run directory.
python "$EXT_ROOT/scripts/prefetch_simkgc.py" --model "$PRETRAINED_MODEL" --attempts 20
mkdir -p "$OUTPUT_DIR"
exec > >(tee "$OUTPUT_DIR/console.log") 2>&1
SCRIPT=train_wn.sh; [[ "$TASK" == FB15k237 ]] && SCRIPT=train_fb.sh
# Preserve all 1024 simultaneous in-batch negatives. Gradient accumulation is not equivalent.
if [[ "$TASK" == NELL995 ]]; then
  bash "$EXT_ROOT/scripts/train_simkgc_nell995.sh" --seed "$SEED" \
    --batch-size 1024 --max-to-keep 1 --pretrained-model "$PRETRAINED_MODEL"
else
  bash "scripts/$SCRIPT" --seed "$SEED" --batch-size 1024 --max-to-keep 1 \
    --pretrained-model "$PRETRAINED_MODEL"
fi
# Official README evaluates model_last. Report this policy, not validation-MRR selection.
bash scripts/eval.sh "$OUTPUT_DIR/model_last.mdl" "$TASK" "$DATA_DIR/valid.txt.json"
bash scripts/eval.sh "$OUTPUT_DIR/model_last.mdl" "$TASK"
