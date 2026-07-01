#!/bin/bash
# =============================================================================
# Title:   Stage B1 — tag agenda-item (TOP) boundaries in transcripts (LLM)
# Purpose: Run scripts/tag_transcript_tops.py over diarised transcripts, inserting
#          <SD-TOP> markers by locating each agenda item in the transcript with
#          gpt-oss-120b. Input  data/transcripts/md (+ data/protocols/md);
#          output data/transcripts/md_top. Skips already-tagged sessions unless
#          OVERWRITE=1, so it is incremental. CPU-only + network (no GPU).
#          Runtime: minutes-to-a-few-hours depending on session count/concurrency.
# Cluster: written for the HPI sc cluster (SLURM). Every "# ADJUST:" line must be
#          changed for a different SLURM cluster.
# Submit:  gpt-oss-120b needs OPENAI_API_KEY, which is NOT in the shared token-free
#          .env. Export it at submit time (inherited via --export=ALL, not clobbered):
#            set -a; . ~/pilotproject-automatic-protocols/.env; set +a
#            sbatch --export=ALL scripts/tag_transcript_tops.sbatch
# =============================================================================

#SBATCH --job-name=tag-tops                   # drives %x in the log names
#SBATCH --account=aisc                         # ADJUST: cluster account
#SBATCH --qos=aisc                             # ADJUST: QOS for the account
#SBATCH --partition=aisc-batch                 # ADJUST: only aisc-batch accepts sbatch (CPU-only job: no --gres)
#SBATCH --constraint=ARCH:X86                  # ADJUST: the uv binary is x86-only
#SBATCH --exclude=gx13v1                       # ADJUST: known-flaky node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                       # ADJUST: CPUs (light; work is LLM/network-bound)
#SBATCH --mem=32G                               # ADJUST: RAM
#SBATCH --time=12:00:00                          # ADJUST: walltime <= 7d partition ceiling
#SBATCH --output=logs/%x_%j.out                 # %x=job-name, %j=job-id
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Token-free shared .env supplies OPENAI_API_BASE + UV_NO_SYNC; the secret
# OPENAI_API_KEY is inherited from the submit environment and NOT overwritten here.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-data/transcripts/md}"
PROTOCOL_DIR="${PROTOCOL_DIR:-data/protocols/md}"
OUT_DIR="${OUT_DIR:-data/transcripts/md_top}"
CONCURRENCY="${CONCURRENCY:-4}"

ARGS=(--transcript-dir "$TRANSCRIPT_DIR" --protocol-dir "$PROTOCOL_DIR"
      --out-dir "$OUT_DIR" --concurrency "$CONCURRENCY")
[ "${OVERWRITE:-0}" = "1" ] && ARGS+=(--overwrite)
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

echo "[$(date -Is)] B1 tag_transcript_tops :: $TRANSCRIPT_DIR + $PROTOCOL_DIR -> $OUT_DIR (concurrency=$CONCURRENCY, overwrite=${OVERWRITE:-0})"
uv run python scripts/tag_transcript_tops.py "${ARGS[@]}"
