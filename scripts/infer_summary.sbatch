#!/bin/bash
#SBATCH --job-name=infer_summary
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc-batch
# Room for gemma-4-31B-it plus long-context (up to 256K) KV cache; device_map=auto
# shards across the GPUs. Override per run: sbatch --gres=gpu:h100:N --mem=XG
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=256G
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
BASE_MODEL="${BASE_MODEL:-google/gemma-4-31B-it}"
ADAPTER="${ADAPTER:-results/lora_adapter}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"   # 65k default; set 0 for full context window
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"  # empty = script default (4096)

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"

adapter_arg=()
[ -d "$ADAPTER" ] && adapter_arg=(--adapter "$ADAPTER")
seqlen_arg=()
[ -n "$MAX_SEQ_LEN" ] && seqlen_arg=(--max-seq-len "$MAX_SEQ_LEN")
newtok_arg=()
[ -n "$MAX_NEW_TOKENS" ] && newtok_arg=(--max-new-tokens "$MAX_NEW_TOKENS")

uv run python scripts/infer_summary.py \
    --input       "$INPUT" \
    --out-dir     "$OUT_DIR" \
    --base-model  "$BASE_MODEL" \
    "${adapter_arg[@]}" \
    "${seqlen_arg[@]}" \
    "${newtok_arg[@]}"
