#!/bin/bash
# =============================================================================
# Title:   Stage A — transcribe and diarise session audio (WhisperX + pyannote)
# Purpose: Run scripts/transcribe.py over every audio path in INPUT_LIST, writing
#          data/transcripts/md/<stem>_Transkript.md with <SD-SPK> speaker tags and
#          timestamps. Sequential; use transcribe_array.sbatch for many files.
#          Needs HF_TOKEN in .env for the gated pyannote diariser.
# Cluster: written for the HPI sc cluster (SLURM). Every "# ADJUST:" line must be
#          changed for a different SLURM cluster.
# Submit:  INPUT_LIST=data/transcripts/manifest_new.txt OUT_DIR=data/transcripts/md DIARIZE=1 sbatch scripts/transcribe.sbatch
# =============================================================================

#SBATCH --job-name=transcribe
#SBATCH --account=aisc                      # ADJUST: cluster account
#SBATCH --qos=aisc                          # ADJUST: QOS for the account
#SBATCH --partition=aisc-batch              # ADJUST: GPU partition that accepts sbatch
#SBATCH --gres=gpu:1                        # ADJUST: any 24 GB+ GPU
#SBATCH --cpus-per-task=4                   # ADJUST: CPUs
#SBATCH --nodes=1
#SBATCH --mem=32G                           # ADJUST: RAM
#SBATCH --time=08:00:00                     # ADJUST: walltime
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
