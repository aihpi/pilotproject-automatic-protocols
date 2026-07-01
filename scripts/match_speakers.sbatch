#!/bin/bash
# =============================================================================
# Title:   Stage B2 — resolve speaker names in TOP-tagged transcripts (LLM)
# Purpose: Run scripts/match_speakers.py over data/transcripts/md_top, replacing
#          SPEAKER_NN with real "Name (Rolle)" and (with --transcript-llm) naming
#          guests absent from the protocol. Output data/transcripts/md_prepared_tllm,
#          per-session maps in data/speaker_maps_tllm, exclusions manifest in
#          data/exclusions/. CPU-only + network (gpt-oss-120b). Concurrency is low
#          by default because the endpoint throttles parallel calls.
#          NOTE: match_speakers.py has no skip-existing; it processes the whole
#          --transcript-dir. For a corpus-wide refresh set OVERWRITE=1; for a delta
#          point TRANSCRIPT_DIR at a directory holding only the new md_top files.
# Cluster: written for the HPI sc cluster (SLURM). Every "# ADJUST:" line must be
#          changed for a different SLURM cluster.
# Submit:  gpt-oss-120b needs OPENAI_API_KEY, which is NOT in the shared token-free
#          .env. Export it at submit time (inherited via --export=ALL, not clobbered):
#            set -a; . ~/pilotproject-automatic-protocols/.env; set +a
#            sbatch --export=ALL scripts/match_speakers.sbatch
# =============================================================================

#SBATCH --job-name=match-speakers             # drives %x in the log names
#SBATCH --account=aisc                         # ADJUST: cluster account
#SBATCH --qos=aisc                             # ADJUST: QOS for the account
#SBATCH --partition=aisc-batch                 # ADJUST: only aisc-batch accepts sbatch (CPU-only job: no --gres)
#SBATCH --constraint=ARCH:X86                  # ADJUST: the uv binary is x86-only
#SBATCH --exclude=gx13v1                       # ADJUST: known-flaky node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                       # ADJUST: CPUs (light; work is LLM/network-bound)
#SBATCH --mem=32G                               # ADJUST: RAM
#SBATCH --time=24:00:00                          # ADJUST: walltime <= 7d partition ceiling
#SBATCH --output=logs/%x_%j.out                 # %x=job-name, %j=job-id
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/exclusions

# Token-free shared .env supplies OPENAI_API_BASE + UV_NO_SYNC; the secret
# OPENAI_API_KEY is inherited from the submit environment and NOT overwritten here.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-data/transcripts/md_top}"
PROTOCOL_DIR="${PROTOCOL_DIR:-data/protocols/md}"
OUT_TRANSCRIPT_DIR="${OUT_TRANSCRIPT_DIR:-data/transcripts/md_prepared_tllm}"
REPORT_DIR="${REPORT_DIR:-data/speaker_maps_tllm}"
EXCLUSIONS_OUT="${EXCLUSIONS_OUT:-data/exclusions/exclusions_tllm.json}"
CONCURRENCY="${CONCURRENCY:-2}"                  # gpt-oss throttles parallel calls

ARGS=(--transcript-dir "$TRANSCRIPT_DIR" --protocol-dir "$PROTOCOL_DIR"
      --out-transcript-dir "$OUT_TRANSCRIPT_DIR" --report-dir "$REPORT_DIR"
      --exclusions-out "$EXCLUSIONS_OUT" --concurrency "$CONCURRENCY")
[ "${TRANSCRIPT_LLM:-1}" = "1" ] && ARGS+=(--transcript-llm)
[ "${OVERWRITE:-0}" = "1" ] && ARGS+=(--overwrite)
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

echo "[$(date -Is)] B2 match_speakers :: $TRANSCRIPT_DIR -> $OUT_TRANSCRIPT_DIR (concurrency=$CONCURRENCY, transcript_llm=${TRANSCRIPT_LLM:-1}, overwrite=${OVERWRITE:-0})"
uv run python scripts/match_speakers.py "${ARGS[@]}"
