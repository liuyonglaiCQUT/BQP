#!/usr/bin/env bash
set -euo pipefail
EXT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXT_ROOT
export HF_HOME="$EXT_ROOT/cache/huggingface"
export TORCH_HOME="$EXT_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$EXT_ROOT/cache/torch_extensions"
export XDG_CACHE_HOME="$EXT_ROOT/cache/xdg"
export PIP_CACHE_DIR="$EXT_ROOT/cache/pip"
export TMPDIR="$EXT_ROOT/tmp"
export WANDB_MODE=disabled
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$EXT_ROOT/logs" "$EXT_ROOT/results" "$EXT_ROOT/figures"
export PYTHONUNBUFFERED=1
cd "$EXT_ROOT"
