#!/bin/bash
# =============================================================================
# Title:   Stage C — build the train/val LoRA dataset from prepared transcripts
# Purpose: Run scripts/build_dataset.py, pairing prepared transcripts to protocols
#          per agenda item and writing chat-format train.jsonl / val.jsonl into a
#          per-dataset subfolder under data/train/. CPU-only; downloads the (gated)
#          gemma tokenizer once. Input data/transcripts/md_prepared_tllm (+
#          data/protocols/md, exclusions in data/exclusions/); output data/train/<name>.
#          Runtime: minutes (tokenisation only).
# Cluster: written for the HPI sc cluster (SLURM). Every "# ADJUST:" line must be
#          changed for a different SLURM cluster.
# Submit:  the gated gemma tokenizer needs HF_TOKEN, which is NOT in the shared
#          token-free .env. Export it at submit time (inherited via --export=ALL):
#            set -a; . ~/pilotproject-automatic-protocols/.env; set +a
#            sbatch --export=ALL scripts/build_dataset.sbatch
# =============================================================================

#SBATCH --job-name=build-dataset              # drives %x in the log names
#SBATCH --account=aisc                         # ADJUST: cluster account
#SBATCH --qos=aisc                             # ADJUST: QOS for the account
#SBATCH --partition=aisc-batch                 # ADJUST: only aisc-batch accepts sbatch (CPU-only job: no --gres)
#SBATCH --constraint=ARCH:X86                  # ADJUST: the uv binary is x86-only
#SBATCH --exclude=gx13v1                       # ADJUST: known-flaky node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                       # ADJUST: CPUs
#SBATCH --mem=48G                               # ADJUST: RAM (tokeniser on long records)
#SBATCH --time=04:00:00                          # ADJUST: walltime <= 7d partition ceiling
#SBATCH --output=logs/%x_%j.out                 # %x=job-name, %j=job-id
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Token-free shared .env supplies UV_NO_SYNC; the secret HF_TOKEN (gated gemma
# tokenizer) is inherited from the submit environment and NOT overwritten here.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-data/transcripts/md_prepared_tllm}"
PROTOCOL_DIR="${PROTOCOL_DIR:-data/protocols/md}"
EXCLUSIONS="${EXCLUSIONS:-data/exclusions/exclusions_tllm_build.json}"
OUT_DIR="${OUT_DIR:-data/train/cap65k}"
GRANULARITY="${GRANULARITY:-per-top}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
HOLDOUT_MANIFEST="${HOLDOUT_MANIFEST:-test/manifest.tsv}"

ARGS=(--transcript-dir "$TRANSCRIPT_DIR" --protocol-dir "$PROTOCOL_DIR"
      --granularity "$GRANULARITY" --max-seq-len "$MAX_SEQ_LEN" --out-dir "$OUT_DIR")
[ -f "$EXCLUSIONS" ] && ARGS+=(--exclusions "$EXCLUSIONS")
[ -f "$HOLDOUT_MANIFEST" ] && ARGS+=(--holdout-manifest "$HOLDOUT_MANIFEST")
[ "${INCLUDE_UNTAGGED:-1}" = "1" ] && ARGS+=(--include-untagged-as-document)
[ "${OVERWRITE:-1}" = "1" ] && ARGS+=(--overwrite)
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

echo "[$(date -Is)] C build_dataset :: $TRANSCRIPT_DIR -> $OUT_DIR (granularity=$GRANULARITY, max_seq_len=$MAX_SEQ_LEN)"
uv run python scripts/build_dataset.py "${ARGS[@]}"
