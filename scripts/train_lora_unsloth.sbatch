#!/bin/bash
#SBATCH --job-name=train_unsloth
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
# Unsloth runs single-GPU: gemma-4-31B-it in 4-bit (~16 GB weights) + Unsloth's
# fused CE/flash-attn fits one 80 GB H100 even at long context. Override per run.
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=ARCH:X86          # uv is x86-only; ARM GPU nodes crash (handoff §7)
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_unsloth_%j.out
#SBATCH --error=logs/train_unsloth_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

set -a
[ -f .env ] && . ./.env
set +a

# Pinned comparison dataset (cap 32768). Smoke: BASE_MODEL=google/gemma-4-E2B-it
# MAX_SEQ_LEN=4096 MAX_STEPS=20
TRAIN_JSONL="${TRAIN_JSONL:-data/train_no_docs/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/train_no_docs/val.jsonl}"
OUT_DIR="${OUT_DIR:-results/unsloth_lora}"
BASE_MODEL="${BASE_MODEL:-google/gemma-4-31B-it}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"
# Triton (Unsloth kernels) JIT-compiles a helper that #includes Python.h.
PYH="$(find "$HOME/.local/share/uv/python" -name Python.h -path '*python3.12*' 2>/dev/null | head -1)"
[ -n "$PYH" ] && export CPATH="$(dirname "$PYH"):${CPATH:-}"

val_arg=()
[ -f "$VAL_JSONL" ] && val_arg=(--val-jsonl "$VAL_JSONL")

# --extra unsloth installs the isolated dependency group (see pyproject.toml).
uv run --extra unsloth python scripts/train_lora_unsloth.py \
    --train-jsonl "$TRAIN_JSONL" \
    "${val_arg[@]}" \
    --base-model  "$BASE_MODEL" \
    --out-dir     "$OUT_DIR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --epochs      "$EPOCHS" \
    --max-steps   "$MAX_STEPS"
