#!/bin/bash
#SBATCH --job-name=train_unsloth
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
#SBATCH --qos=aisc                     # AISC QOS = faster scheduling
# Unsloth runs single-GPU: gemma-4-31B-it in 4-bit (~16 GB weights) + Unsloth's
# fused CE/flash-attn fits one 80 GB H100 even at long context. Override per run.
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=ARCH:X86          # uv is x86-only; ARM GPU nodes crash (handoff §7)
#SBATCH --exclude=gx13v1               # flaky node (nvlink/hw faults); see handoff §7
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

# Default dataset = 65k cap. Smoke: BASE_MODEL=google/gemma-4-E2B-it
# MAX_SEQ_LEN=4096 MAX_STEPS=20
TRAIN_JSONL="${TRAIN_JSONL:-data/train_no_docs_cap65k/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/train_no_docs_cap65k/val.jsonl}"
OUT_DIR="${OUT_DIR:-}"             # empty = script auto-names results/YYYYMMDD-HHMMSS
BASE_MODEL="${BASE_MODEL:-google/gemma-4-31B-it}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"   # 65k cap (was 32768)
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
# Which Unsloth trainer to run: the bespoke one (default) or the Kaggle-faithful v2.
# v2's --epochs default is 1; pass EPOCHS explicitly to override either way.
SCRIPT="${SCRIPT:-scripts/train_lora_unsloth.py}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"
# Triton (Unsloth kernels) JIT-compiles a helper that #includes Python.h.
PYH="$(find "$HOME/.local/share/uv/python" -name Python.h -path '*python3.12*' 2>/dev/null | head -1)"
[ -n "$PYH" ] && export CPATH="$(dirname "$PYH"):${CPATH:-}"

val_arg=()
[ -f "$VAL_JSONL" ] && val_arg=(--val-jsonl "$VAL_JSONL")

# Standalone venv (NOT the project env): Unsloth's deps conflict with the base
# pyproject pins (torch>=2.11/numpy>=2.4.4/liger), so each alt-framework gets its
# own venv built outside the project resolver. Build once on a run node:
#   uv venv .venv-unsloth --python 3.12
#   uv pip install --python .venv-unsloth unsloth trl peft datasets bitsandbytes tensorboard
VENV="${VENV:-.venv-unsloth}"
[ -x "$VENV/bin/python" ] || { echo "missing $VENV (see comment above to build it)"; exit 1; }

out_arg=()
[ -n "$OUT_DIR" ] && out_arg=(--out-dir "$OUT_DIR")

# NO4BIT=1 -> bf16 LoRA (no quantization); default is 4-bit QLoRA.
bit_arg=()
[ "${NO4BIT:-0}" = "1" ] && bit_arg=(--no-4bit)

echo "[$(date -Is)] SCRIPT=$SCRIPT VENV=$VENV TRAIN=$TRAIN_JSONL MAXSEQ=$MAX_SEQ_LEN EPOCHS=$EPOCHS"
"$VENV/bin/python" "$SCRIPT" \
    --train-jsonl "$TRAIN_JSONL" \
    "${val_arg[@]}" \
    --base-model  "$BASE_MODEL" \
    "${out_arg[@]}" \
    --max-seq-len "$MAX_SEQ_LEN" \
    "${bit_arg[@]}" \
    --epochs      "$EPOCHS" \
    --max-steps   "$MAX_STEPS"
