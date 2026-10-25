#!/usr/bin/env bash
set -e
set -x

TASK=NELL995
REPO="${EXT_ROOT}/repos/SimKGC"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
: "${DATA_DIR:?DATA_DIR must be set}"

cd "$REPO"
python -u main.py \
  --model-dir "$OUTPUT_DIR" \
  --pretrained-model bert-base-uncased \
  --pooling mean \
  --lr 1e-5 \
  --use-link-graph \
  --train-path "$DATA_DIR/train.txt.json" \
  --valid-path "$DATA_DIR/valid.txt.json" \
  --task "$TASK" \
  --batch-size 1024 \
  --print-freq 20 \
  --additive-margin 0.02 \
  --use-amp \
  --use-self-negative \
  --finetune-t \
  --pre-batch 2 \
  --epochs 10 \
  --workers 4 \
  --max-to-keep 5 "$@"
