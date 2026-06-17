#!/bin/bash
#SBATCH --job-name=train_fsdp
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
# Sharded across multiple H100 (FSDP/ZeRO-3). 4 is a good start for 31B.
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=ARCH:X86
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
TRAIN_JSONL="${TRAIN_JSONL:-data/train_no_docs/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/train_no_docs/val.jsonl}"
OUT_DIR="${OUT_DIR:-results/fsdp_lora}"
BASE_MODEL="${BASE_MODEL:-google/gemma-4-31B-it}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
NGPU="${NGPU:-4}"
# BACKEND=fsdp (configs/fsdp.yaml) or BACKEND=zero3 (DeepSpeed configs/zero3.json)
BACKEND="${BACKEND:-fsdp}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"

val_arg=()
[ -f "$VAL_JSONL" ] && val_arg=(--val-jsonl "$VAL_JSONL")

if [ "$BACKEND" = "zero3" ]; then
  launch=(--use_deepspeed --deepspeed_config_file configs/zero3.json --num_processes "$NGPU")
else
  launch=(--config_file configs/fsdp.yaml --num_processes "$NGPU")
fi

uv run accelerate launch "${launch[@]}" \
    scripts/train_lora_fsdp.py \
    --train-jsonl "$TRAIN_JSONL" \
    "${val_arg[@]}" \
    --base-model  "$BASE_MODEL" \
    --out-dir     "$OUT_DIR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --epochs      "$EPOCHS" \
    --max-steps   "$MAX_STEPS"
