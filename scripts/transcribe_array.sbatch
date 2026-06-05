#!/bin/bash
#SBATCH --job-name=transcribe_array
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc-batch
#SBATCH --gres=gpu:1                         # one GPU per array task (one file per task)
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/transcribe_%A_%a.out   # %A=array job id, %a=task index
#SBATCH --error=logs/transcribe_%A_%a.err
#SBATCH --requeue                            # let SLURM requeue on node failure/preemption
#
# Parallel per-file transcription via a SLURM job array. Each task takes one GPU
# (possibly on a different node — the tasks are independent) and transcribes its
# shard of the manifest: entries [SLURM_ARRAY_TASK_ID :: NUM_SHARDS].
#
# Launch (array size + concurrency are set on the CLI, not here):
#   L=$(grep -cve '^[[:space:]]*$' manifest.txt)        # number of files
#   INPUT_LIST=manifest.txt OUT_DIR=… DIARIZE=1 NUM_SHARDS=$L \
#       sbatch --array=0-$((L-1))%4 scripts/transcribe_array.sbatch
# %4 caps concurrency at 4 → up to 4 GPUs run in parallel.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load HF_TOKEN etc. from .env (needed for the gated pyannote diariser).
if [ -f .env ]; then set -a; . ./.env; set +a; fi

: "${INPUT_LIST:?Set INPUT_LIST=path/to/manifest.txt (audio or .npy paths)}"
OUT_DIR="${OUT_DIR:-data/transcripts/md}"
LANG_CODE="${LANG_CODE:-de}"
MODEL="${MODEL:-large-v3}"
DIARIZE="${DIARIZE:-0}"               # set DIARIZE=1 to run pyannote speaker diarisation
NUM_SPEAKERS="${NUM_SPEAKERS:-}"      # optional exact speaker count
MAX_TRIES="${MAX_TRIES:-3}"           # retries for transient GPU-grab failures
# Default NUM_SHARDS to the array size if the caller didn't set it explicitly.
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

export PATH="$HOME/.local/bin:$PATH"

ARGS=(--input-list "$INPUT_LIST" --out-dir "$OUT_DIR" --model "$MODEL" \
      --language "$LANG_CODE" \
      --num-shards "$NUM_SHARDS" --shard-index "${SLURM_ARRAY_TASK_ID:-0}")
if [ "$DIARIZE" = "1" ]; then
    ARGS+=(--diarize)
    [ -n "$NUM_SPEAKERS" ] && ARGS+=(--num-speakers "$NUM_SPEAKERS")
fi

echo "task ${SLURM_ARRAY_TASK_ID:-0}/${NUM_SHARDS} on $(hostname); GPU=${CUDA_VISIBLE_DEVICES:-?}"

# Self-heal transient GPU-grab failures (cudaErrorDevicesUnavailable): when an array
# task lands on a node whose just-freed GPU is still being released by the driver,
# context creation reports "CUDA-capable device(s) is/are busy or unavailable". The GPU
# usually frees within seconds, so retry with backoff. Only retry on that signature —
# genuine errors (bad file, OOM) fail fast. transcribe.py skips already-written .md, so
# a retry resumes cleanly.
ATTEMPT_ERR="logs/transcribe_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}.attempt"
TRANSIENT='busy or unavailable|CUDA-capable device|cudaErrorDevicesUnavailable'
for try in $(seq 1 "$MAX_TRIES"); do
    echo ">>> attempt $try/$MAX_TRIES"
    rc=0
    uv run python scripts/transcribe.py "${ARGS[@]}" 2>"$ATTEMPT_ERR" || rc=$?
    cat "$ATTEMPT_ERR" >&2                    # propagate to the SLURM .err log
    if [ "$rc" -eq 0 ]; then
        echo ">>> attempt $try succeeded"
        exit 0
    fi
    if grep -qiE "$TRANSIENT" "$ATTEMPT_ERR"; then transient=1; else transient=0; fi
    if [ "$transient" = 1 ] && [ "$try" -lt "$MAX_TRIES" ]; then
        backoff=$((30 * try))
        echo ">>> transient GPU error (rc=$rc); retrying in ${backoff}s on $(hostname)" >&2
        nvidia-smi >&2 2>/dev/null || true
        sleep "$backoff"
        continue
    fi
    if [ "$transient" = 1 ]; then
        echo ">>> gave up after $MAX_TRIES transient-GPU failures — node $(hostname) likely" \
             "has a stuck/unavailable GPU; resubmit with: sbatch --exclude=$(hostname) …" >&2
    else
        echo ">>> attempt $try failed (rc=$rc); non-transient error — giving up" >&2
    fi
    exit "$rc"
done
exit 1
