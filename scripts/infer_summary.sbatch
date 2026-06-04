#!/bin/bash
#SBATCH --job-name=infer_summary
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc-batch
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/infer_summary_%j.out
#SBATCH --error=logs/infer_summary_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load local secrets (HF_TOKEN, optional HF_HOME) if a .env is present.
set -a
[ -f .env ] && . ./.env
set +a

: "${INPUT:?Set INPUT=path/to/transcript_or_dir}"
OUT_DIR="${OUT_DIR:-results/summaries}"
BASE_MODEL="${BASE_MODEL:-google/gemma-4-E2B-it}"
ADAPTER="${ADAPTER:-results/lora_adapter}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"

adapter_arg=()
[ -d "$ADAPTER" ] && adapter_arg=(--adapter "$ADAPTER")

uv run python scripts/infer_summary.py \
    --input       "$INPUT" \
    --out-dir     "$OUT_DIR" \
    --base-model  "$BASE_MODEL" \
    "${adapter_arg[@]}"
