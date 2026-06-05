#!/bin/bash
#SBATCH --job-name=transcribe
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/transcribe_%j.out
#SBATCH --error=logs/transcribe_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load HF_TOKEN etc. from .env (needed for the gated pyannote diariser).
if [ -f .env ]; then set -a; . ./.env; set +a; fi

: "${INPUT_LIST:?Set INPUT_LIST=path/to/manifest.txt (audio or .npy paths)}"
OUT_DIR="${OUT_DIR:-data/transcripts/md}"
LANG_CODE="${LANG_CODE:-de}"
MODEL="${MODEL:-large-v3}"
DIARIZE="${DIARIZE:-0}"          # set DIARIZE=1 to run pyannote speaker diarisation
NUM_SPEAKERS="${NUM_SPEAKERS:-}" # optional exact speaker count

export PATH="$HOME/.local/bin:$PATH"

ARGS=(--input-list "$INPUT_LIST" --out-dir "$OUT_DIR" --model "$MODEL" --language "$LANG_CODE")
if [ "$DIARIZE" = "1" ]; then
    ARGS+=(--diarize)
    [ -n "$NUM_SPEAKERS" ] && ARGS+=(--num-speakers "$NUM_SPEAKERS")
fi

uv run python scripts/transcribe.py "${ARGS[@]}"
