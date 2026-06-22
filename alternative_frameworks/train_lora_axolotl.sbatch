#!/bin/bash
#SBATCH --job-name=train_axolotl
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
#SBATCH --qos=aisc
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=ARCH:X86
#SBATCH --exclude=gx13v1
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

CONFIG="${CONFIG:-alternative_frameworks/axolotl/gemma4_qlora.yml}"
NGPU="${NGPU:-4}"
# Timestamped run folder (matches the YYYYMMDD-HHMMSS convention); override the
# yaml's output_dir on the CLI. Axolotl writes its adapter + config there.
OUT_DIR="${OUT_DIR:-results/$(date +%Y%m%d-%H%M%S)}"
# Smoke (override Axolotl config keys on the CLI):
#   OVERRIDES="base_model=google/gemma-4-E2B-it sequence_len=4096 max_steps=20"
OVERRIDES="${OVERRIDES:-}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"
PYH="$(find "$HOME/.local/share/uv/python" -name Python.h -path '*python3.12*' 2>/dev/null | head -1)"
[ -n "$PYH" ] && export CPATH="$(dirname "$PYH"):${CPATH:-}"

# Standalone venv (axolotl pins its own transformers/trl, incompatible with the
# base stack). Build once on a run node:
#   uv venv .venv-axolotl --python 3.12 && uv pip install --python .venv-axolotl axolotl
VENV="${VENV:-.venv-axolotl}"
[ -x "$VENV/bin/accelerate" ] || { echo "missing $VENV (see comment above to build it)"; exit 1; }

"$VENV/bin/accelerate" launch --num_processes "$NGPU" \
    -m axolotl.cli.train "$CONFIG" output_dir="$OUT_DIR" $OVERRIDES
