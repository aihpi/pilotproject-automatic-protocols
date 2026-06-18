#!/bin/bash
#SBATCH --job-name=train_keras
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
#SBATCH --qos=aisc
# Minimal = 1 GPU; for a large preset add --model-parallel and bump GPUs (e.g. 4).
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=ARCH:X86
#SBATCH --exclude=gx13v1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_keras_%j.out
#SBATCH --error=logs/train_keras_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

set -a
[ -f .env ] && . ./.env
set +a

# Smoke: PRESET=gemma3_instruct_1b MAX_SEQ_LEN=4096 LIMIT=20
TRAIN_JSONL="${TRAIN_JSONL:-data/train_no_docs/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/train_no_docs/val.jsonl}"
OUT_DIR="${OUT_DIR:-}"             # empty = script auto-names results/YYYYMMDD-HHMMSS
PRESET="${PRESET:-hf://google/gemma-3-1b-it}"   # HF converter route (no Kaggle auth)
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
EPOCHS="${EPOCHS:-3}"
LIMIT="${LIMIT:-}"
MODEL_PARALLEL="${MODEL_PARALLEL:-0}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"
export KERAS_BACKEND=jax
# KerasHub may pull Kaggle-hosted presets; HF presets use HF_HOME. Keep gated tokens in .env.

val_arg=()
[ -f "$VAL_JSONL" ] && val_arg=(--val-jsonl "$VAL_JSONL")
limit_arg=()
[ -n "$LIMIT" ] && limit_arg=(--limit "$LIMIT")
mp_arg=()
[ "$MODEL_PARALLEL" = "1" ] && mp_arg=(--model-parallel)
out_arg=()
[ -n "$OUT_DIR" ] && out_arg=(--out-dir "$OUT_DIR")

# Standalone venv (JAX/Keras conflict with the torch stack). Build once on a run node:
#   uv venv .venv-keras --python 3.12
#   uv pip install --python .venv-keras keras keras-hub "jax[cuda12]"
VENV="${VENV:-.venv-keras}"
[ -x "$VENV/bin/python" ] || { echo "missing $VENV (see comment above to build it)"; exit 1; }

"$VENV/bin/python" scripts/train_lora_keras.py \
    --train-jsonl "$TRAIN_JSONL" \
    "${val_arg[@]}" \
    --preset      "$PRESET" \
    "${out_arg[@]}" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --epochs      "$EPOCHS" \
    "${limit_arg[@]}" \
    "${mp_arg[@]}"
