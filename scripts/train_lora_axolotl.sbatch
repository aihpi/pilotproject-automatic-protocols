#!/bin/bash
#SBATCH --job-name=train_axolotl
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=ARCH:X86
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_axolotl_%j.out
#SBATCH --error=logs/train_axolotl_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

set -a
[ -f .env ] && . ./.env
set +a

CONFIG="${CONFIG:-axolotl/gemma4_qlora.yml}"
NGPU="${NGPU:-4}"
# Smoke (override Axolotl config keys on the CLI):
#   OVERRIDES="base_model=google/gemma-4-E2B-it sequence_len=4096 max_steps=20"
OVERRIDES="${OVERRIDES:-}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"
PYH="$(find "$HOME/.local/share/uv/python" -name Python.h -path '*python3.12*' 2>/dev/null | head -1)"
[ -n "$PYH" ] && export CPATH="$(dirname "$PYH"):${CPATH:-}"

# Isolated env (axolotl pins its own transformers/trl) -> separate extra.
uv run --extra axolotl accelerate launch --num_processes "$NGPU" \
    -m axolotl.cli.train "$CONFIG" $OVERRIDES
