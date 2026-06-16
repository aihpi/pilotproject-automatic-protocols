#!/bin/bash
#SBATCH --job-name=train_lora
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc-batch
# 16-bit LoRA on gemma-4-31B-it (~70 GB weights) plus long, un-truncated
# sequences (max-seq-len = context window) exceeds one H100; request several so
# device_map=auto can shard. Override per run: sbatch --gres=gpu:h100:N --mem=XG
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
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
OUT_DIR="${OUT_DIR:-}"             # empty = auto-name (results/YYYYMMDD_XX_lora; smoke -> smoke_lora)
BASE_MODEL="${BASE_MODEL:-google/gemma-4-31B-it}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"     # empty = auto (model context window)
BITS="${BITS:-16}"                 # 16 = real bf16 LoRA; smoke test: BITS=4 MAX_STEPS=5
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"   # set e.g. 5 for a quick smoke test
# Packing is OFF by default: with the default sdpa attention, packing flattens
# several samples into one sequence and they can attend across each other
# (cross-contamination). Only set PACKING=1 together with a flash-attention
# build + --attn flash_attention_2, which masks packed samples correctly.
PACKING="${PACKING:-0}"
ATTN="${ATTN:-}"               # e.g. flash_attention_2 (needs flash-attn) for long sequences

export PATH="$HOME/.local/bin:$PATH"
# Keep large model downloads off the home quota (shared project storage).
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"

# Liger kernels use Triton, which JIT-compiles a small CUDA helper that #includes
# Python.h. The cluster's system python3.12 has no dev headers, so expose the
# uv-managed standalone CPython 3.12 headers to gcc via CPATH.
PYH="$(find "$HOME/.local/share/uv/python" -name Python.h -path '*python3.12*' 2>/dev/null | head -1)"
[ -n "$PYH" ] && export CPATH="$(dirname "$PYH"):${CPATH:-}"

val_arg=()
[ -f "$VAL_JSONL" ] && val_arg=(--val-jsonl "$VAL_JSONL")

pack_arg=()
[ "$PACKING" = "1" ] && pack_arg=(--packing)

out_arg=()
[ -n "$OUT_DIR" ] && out_arg=(--out-dir "$OUT_DIR")

seqlen_arg=()
[ -n "$MAX_SEQ_LEN" ] && seqlen_arg=(--max-seq-len "$MAX_SEQ_LEN")

attn_arg=()
[ -n "$ATTN" ] && attn_arg=(--attn "$ATTN")

liger_arg=()
[ "${USE_LIGER:-0}" = "1" ] && liger_arg=(--use-liger)

uv run python scripts/train_lora.py \
    --train-jsonl  "$TRAIN_JSONL" \
    "${val_arg[@]}" \
    --base-model   "$BASE_MODEL" \
    "${out_arg[@]}" \
    --bits         "$BITS" \
    "${seqlen_arg[@]}" \
    "${attn_arg[@]}" \
    "${liger_arg[@]}" \
    --epochs       "$EPOCHS" \
    --max-steps    "$MAX_STEPS" \
    "${pack_arg[@]}"
