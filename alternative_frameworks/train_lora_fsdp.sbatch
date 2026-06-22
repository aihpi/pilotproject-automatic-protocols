#!/bin/bash
#SBATCH --job-name=train_fsdp
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
#SBATCH --qos=aisc
# Sharded across multiple H100 (FSDP/ZeRO-3). 4 is a good start for 31B.
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=ARCH:X86
#SBATCH --exclude=gx13v1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_fsdp_%j.out
#SBATCH --error=logs/train_fsdp_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

set -a
[ -f .env ] && . ./.env
set +a

# Smoke: BASE_MODEL=google/gemma-4-E2B-it MAX_SEQ_LEN=4096 MAX_STEPS=20 NGPU=2
TRAIN_JSONL="${TRAIN_JSONL:-data/train_no_docs_cap65k/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/train_no_docs_cap65k/val.jsonl}"
OUT_DIR="${OUT_DIR:-}"             # empty = script auto-names results/YYYYMMDD-HHMMSS
BASE_MODEL="${BASE_MODEL:-google/gemma-4-31B-it}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"   # 65k cap (was 16384); needs CCE=1 at this length
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
NGPU="${NGPU:-4}"
# BACKEND=fsdp (alternative_frameworks/configs/fsdp.yaml) or BACKEND=zero3 (DeepSpeed alternative_frameworks/configs/zero3.json)
BACKEND="${BACKEND:-fsdp}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"
# cut-cross-entropy (--cce / CCE=1) uses Triton, which JIT-compiles a launcher that
# #includes Python.h. The cluster python lacks dev headers — expose the uv cpython
# headers to gcc via CPATH (same as train_lora.sbatch / the unsloth sbatch).
PYH="$(find "$HOME/.local/share/uv/python" -name Python.h -path '*python3.12*' 2>/dev/null | head -1)"
[ -n "$PYH" ] && export CPATH="$(dirname "$PYH"):${CPATH:-}"

val_arg=()
[ -f "$VAL_JSONL" ] && val_arg=(--val-jsonl "$VAL_JSONL")
out_arg=()
[ -n "$OUT_DIR" ] && out_arg=(--out-dir "$OUT_DIR")

# FSDP reuses the proven base training venv (transformers/peft/trl/accelerate) of
# the MAIN checkout — no separate build. For the ZeRO-3 backend, that venv also
# needs deepspeed: uv pip install --python <BASE_VENV> deepspeed
BASE_VENV="${BASE_VENV:-/sc/home/hanno.mueller/pilotproject-automatic-protocols/.venv}"
[ -x "$BASE_VENV/bin/accelerate" ] || { echo "missing accelerate in $BASE_VENV"; exit 1; }

if [ "$BACKEND" = "zero3" ]; then
  launch=(--use_deepspeed --deepspeed_config_file alternative_frameworks/configs/zero3.json --num_processes "$NGPU")
else
  launch=(--config_file alternative_frameworks/configs/fsdp.yaml --num_processes "$NGPU")
fi

# CCE=1 -> cut-cross-entropy loss (required for cap 32768 / uncapped: FSDP shards
# weights but not the seq×vocab logits term).
cce_arg=()
[ "${CCE:-0}" = "1" ] && cce_arg=(--cce)

"$BASE_VENV/bin/accelerate" launch "${launch[@]}" \
    alternative_frameworks/train_lora_fsdp.py \
    --train-jsonl "$TRAIN_JSONL" \
    "${val_arg[@]}" \
    --base-model  "$BASE_MODEL" \
    "${out_arg[@]}" \
    --max-seq-len "$MAX_SEQ_LEN" \
    "${cce_arg[@]}" \
    --epochs      "$EPOCHS" \
    --max-steps   "$MAX_STEPS"
