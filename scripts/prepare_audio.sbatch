#!/bin/bash
#SBATCH --job-name=prepare_audio
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc-batch                    # AISC partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepare_audio_%j.out
#SBATCH --error=logs/prepare_audio_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

: "${INPUT_LIST:?Set INPUT_LIST=path/to/mp4_manifest.txt}"
OUT_DIR="${OUT_DIR:-data/transcripts/spectrograms}"

export PATH="$HOME/.local/bin:$PATH"

uv run python scripts/prepare_audio.py \
    --input-list "$INPUT_LIST" \
    --out-dir    "$OUT_DIR" \
    --workers    "${SLURM_CPUS_PER_TASK:-8}"
