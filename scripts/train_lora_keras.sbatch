#!/bin/bash
#SBATCH --job-name=train_keras
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
# Minimal = 1 GPU; for a large preset add --model-parallel and bump GPUs (e.g. 4).
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=ARCH:X86
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
OUT_DIR="${OUT_DIR:-results/keras_lora}"
PRESET="${PRESET:-gemma3_instruct_1b}"
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

# Isolated env: JAX/Keras conflict with the torch stack -> separate extra.
uv run --extra keras python scripts/train_lora_keras.py \
    --train-jsonl "$TRAIN_JSONL" \
    "${val_arg[@]}" \
    --preset      "$PRESET" \
    --out-dir     "$OUT_DIR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --epochs      "$EPOCHS" \
    "${limit_arg[@]}" \
    "${mp_arg[@]}"
