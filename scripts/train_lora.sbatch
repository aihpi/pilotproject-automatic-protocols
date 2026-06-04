#!/bin/bash
#SBATCH --job-name=train_lora
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc-batch
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_lora_%j.out
#SBATCH --error=logs/train_lora_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load local secrets (HF_TOKEN, optional HF_HOME) if a .env is present.
set -a
[ -f .env ] && . ./.env
set +a

: "${TRAIN_JSONL:?Set TRAIN_JSONL=path/to/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/train/val.jsonl}"
OUT_DIR="${OUT_DIR:-results/lora_adapter}"
BASE_MODEL="${BASE_MODEL:-google/gemma-4-E2B-it}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
BITS="${BITS:-4}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"   # set e.g. 5 for a quick smoke test

export PATH="$HOME/.local/bin:$PATH"
# Keep large model downloads off the home quota (shared project storage).
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"

val_arg=()
[ -f "$VAL_JSONL" ] && val_arg=(--val-jsonl "$VAL_JSONL")

uv run python scripts/train_lora.py \
    --train-jsonl  "$TRAIN_JSONL" \
    "${val_arg[@]}" \
    --base-model   "$BASE_MODEL" \
    --out-dir      "$OUT_DIR" \
    --bits         "$BITS" \
    --max-seq-len  "$MAX_SEQ_LEN" \
    --epochs       "$EPOCHS" \
    --max-steps    "$MAX_STEPS" \
    --packing
