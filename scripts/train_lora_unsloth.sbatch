#!/bin/bash
# =============================================================================
# Title:   Stage D — QLoRA fine-tune of gemma-4-31B-it with Unsloth (single GPU)
# Purpose: Run scripts/train_lora_unsloth.py from the standalone .venv-unsloth on one
#          80 GB H100: 4-bit base, LoRA r8/alpha8, 3 epochs with early stopping.
#          Writes results/YYYYMMDD-HHMMSS/ (adapter, tokenizer, train_log.md) or
#          OUT_DIR. Input data/train/<name>/{train,val}.jsonl from stage C.
#          Runtime: about 20 h for the cap48k set (1115 records, cap 49152).
# Cluster: written for the HPI sc cluster (SLURM). Every "# ADJUST:" line must be
#          changed for a different SLURM cluster.
# Submit:  TRAIN_JSONL=data/train/cap48k/train.jsonl VAL_JSONL=data/train/cap48k/val.jsonl MAX_SEQ_LEN=49152 OUT_DIR=results/$(date +%Y%m%d)-31b_cap48k sbatch scripts/train_lora_unsloth.sbatch
# =============================================================================

#SBATCH --job-name=train_unsloth
#SBATCH --account=aisc                 # ADJUST: cluster account
#SBATCH --qos=aisc                     # ADJUST: QOS for the account
#SBATCH --partition=aisc-batch         # ADJUST: GPU partition that accepts sbatch
# Unsloth runs single-GPU: gemma-4-31B-it in 4-bit (~16 GB weights) plus Unsloth's
# fused cross-entropy fits one 80 GB H100 up to a 49152-token cap (65536 OOMs).
#SBATCH --gres=gpu:h100:1              # ADJUST: one 80 GB GPU
#SBATCH --constraint=ARCH:X86          # ADJUST: the uv binary is x86-only
#SBATCH --exclude=gx13v1               # ADJUST: known-flaky node
#SBATCH --cpus-per-task=16             # ADJUST: CPUs
#SBATCH --nodes=1
#SBATCH --mem=128G                     # ADJUST: RAM
#SBATCH --time=48:00:00                # ADJUST: walltime (cap48k needs ~20 h)
#SBATCH --output=logs/train_unsloth_%j.out
#SBATCH --error=logs/train_unsloth_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

set -a
[ -f .env ] && . ./.env
set +a

# Default dataset = the production cap48k set. Smoke test on a small base:
# BASE_MODEL=google/gemma-4-E2B-it MAX_SEQ_LEN=4096 MAX_STEPS=20
TRAIN_JSONL="${TRAIN_JSONL:-data/train/cap48k/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/train/cap48k/val.jsonl}"
OUT_DIR="${OUT_DIR:-}"             # empty = script auto-names results/YYYYMMDD-HHMMSS
BASE_MODEL="${BASE_MODEL:-google/gemma-4-31B-it}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-49152}"   # records longer than this were excluded at build
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
# The trainer's own --epochs default is 1; this sbatch defaults EPOCHS=3 above
# (best-checkpoint selection plus early stopping guard against overfitting).
SCRIPT="${SCRIPT:-scripts/train_lora_unsloth.py}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/hf-cache}"   # override in .env for shared storage
# Triton (Unsloth kernels) JIT-compiles a helper that #includes Python.h.
PYH="$(find "$HOME/.local/share/uv/python" -name Python.h -path '*python3.12*' 2>/dev/null | head -1)"
[ -n "$PYH" ] && export CPATH="$(dirname "$PYH"):${CPATH:-}"

val_arg=()
[ -f "$VAL_JSONL" ] && val_arg=(--val-jsonl "$VAL_JSONL")

# Standalone venv (NOT the project env): Unsloth's deps conflict with the base
# pyproject pins (torch>=2.11/numpy>=2.4.4), so it is built outside the project
# resolver. Build once on a run node:
#   uv venv .venv-unsloth --python 3.12
#   uv pip install --python .venv-unsloth unsloth trl peft datasets bitsandbytes tensorboard
VENV="${VENV:-.venv-unsloth}"
[ -x "$VENV/bin/python" ] || { echo "missing $VENV (see comment above to build it)"; exit 1; }

out_arg=()
[ -n "$OUT_DIR" ] && out_arg=(--out-dir "$OUT_DIR")

# NO4BIT=1 -> bf16 LoRA (no quantization); default is 4-bit QLoRA.
bit_arg=()
[ "${NO4BIT:-0}" = "1" ] && bit_arg=(--no-4bit)

# Optional hyperparameter overrides for sweeps. Omit -> the script's own defaults
# (r8/alpha8/dropout0, lr 2e-4, bs1/ga4, warmup 5, wd 0.001, early-stop patience 3).
sweep_args=()
[ -n "${LORA_R:-}" ]       && sweep_args+=(--lora-r "$LORA_R")
[ -n "${LORA_ALPHA:-}" ]   && sweep_args+=(--lora-alpha "$LORA_ALPHA")
[ -n "${LORA_DROPOUT:-}" ] && sweep_args+=(--lora-dropout "$LORA_DROPOUT")
[ -n "${LR:-}" ]           && sweep_args+=(--lr "$LR")
[ -n "${BATCH_SIZE:-}" ]   && sweep_args+=(--batch-size "$BATCH_SIZE")
[ -n "${GRAD_ACCUM:-}" ]   && sweep_args+=(--grad-accum "$GRAD_ACCUM")
[ -n "${WARMUP_STEPS:-}" ] && sweep_args+=(--warmup-steps "$WARMUP_STEPS")
[ -n "${WEIGHT_DECAY:-}" ] && sweep_args+=(--weight-decay "$WEIGHT_DECAY")
[ -n "${ES_PATIENCE:-}" ]  && sweep_args+=(--early-stopping-patience "$ES_PATIENCE")

echo "[$(date -Is)] SCRIPT=$SCRIPT VENV=$VENV TRAIN=$TRAIN_JSONL MAXSEQ=$MAX_SEQ_LEN EPOCHS=$EPOCHS ${sweep_args[*]}"
"$VENV/bin/python" "$SCRIPT" \
    --train-jsonl "$TRAIN_JSONL" \
    "${val_arg[@]}" \
    --base-model  "$BASE_MODEL" \
    "${out_arg[@]}" \
    --max-seq-len "$MAX_SEQ_LEN" \
    "${bit_arg[@]}" \
    --epochs      "$EPOCHS" \
    --max-steps   "$MAX_STEPS" \
    "${sweep_args[@]}"
