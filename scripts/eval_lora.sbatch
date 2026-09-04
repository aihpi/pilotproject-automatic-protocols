#!/bin/bash
# =============================================================================
# Title:   Stage E — summarise an examples directory with one LoRA adapter
# Purpose: Run scripts/eval_lora.py under the framework's venv: load base model and
#          adapter once, write one protocol per (example x decode preset) into
#          RUN_DIR. An example is a folder holding <stem>_Transkript.md and,
#          optionally, a gold <stem>_Protokoll.md for eval_report.py. Default
#          examples: test/ (the held-out set).
# Cluster: written for the HPI sc cluster (SLURM). Every "# ADJUST:" line must be
#          changed for a different SLURM cluster.
# Submit:  FRAMEWORK=unsloth ADAPTER=results/20260902-31b_cap48k ADAPTER_ID=cap48k MAX_SEQ_LEN=49152 RUN_DIR=data/test/$(date +%Y%m%d-%H%M%S) sbatch scripts/eval_lora.sbatch
#          New transcripts without gold: add EXAMPLES_DIR=path/to/examples
# =============================================================================

#SBATCH --job-name=eval_lora
#SBATCH --account=aisc                 # ADJUST: cluster account
#SBATCH --qos=aisc                     # ADJUST: QOS for the account
#SBATCH --partition=aisc-batch         # ADJUST: GPU partition that accepts sbatch
#SBATCH --gres=gpu:h100:1              # ADJUST: one 80 GB GPU serves the 31B in 4-bit
#SBATCH --constraint=ARCH:X86          # ADJUST: the uv binary is x86-only
#SBATCH --exclude=gx13v1               # ADJUST: known-flaky node
#SBATCH --cpus-per-task=16             # ADJUST: CPUs
#SBATCH --nodes=1
#SBATCH --mem=128G                     # ADJUST: RAM
#SBATCH --time=08:00:00                # ADJUST: walltime
#SBATCH --output=logs/eval_lora_%j.out
#SBATCH --error=logs/eval_lora_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

set -a
[ -f .env ] && . ./.env
set +a

: "${FRAMEWORK:?Set FRAMEWORK=peft|unsloth}"
: "${ADAPTER:?Set ADAPTER=path/to/adapter}"
: "${ADAPTER_ID:?Set ADAPTER_ID=short_id}"
BASE_MODEL="${BASE_MODEL:-}"           # empty = read from adapter_config.json
BITS="${BITS:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-49152}"
GRAN="${GRAN:-per-top}"
DECODES="${DECODES:-baseline,antirep}"
ONLY="${ONLY:-}"
EXAMPLES_DIR="${EXAMPLES_DIR:-test}"
RUN_DIR="${RUN_DIR:-}"                  # empty = eval_lora.py picks data/test/<timestamp>
if [ "$FRAMEWORK" = "unsloth" ]; then
  PY="${PY:-.venv-unsloth/bin/python}"   # standalone venv, see train_lora_unsloth.sbatch
else
  PY="${PY:-uv run python}"
fi

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/hf-cache}"   # override in .env for shared storage

# Unsloth's Triton kernels JIT-compile a helper that #includes Python.h.
if [ "$FRAMEWORK" = "unsloth" ]; then
  PYH="$(find "$HOME/.local/share/uv/python" -name Python.h -path '*python3.12*' 2>/dev/null | head -1)"
  [ -n "$PYH" ] && export CPATH="$(dirname "$PYH"):${CPATH:-}"
fi

opt=()
[ -n "$BASE_MODEL" ] && opt+=(--base-model "$BASE_MODEL")
[ -n "$ONLY" ]       && opt+=(--only "$ONLY")
[ -n "$RUN_DIR" ]    && opt+=(--run-dir "$RUN_DIR")

echo "[$(date -Is)] FRAMEWORK=$FRAMEWORK ADAPTER=$ADAPTER ID=$ADAPTER_ID BITS=$BITS MAXSEQ=$MAX_SEQ_LEN EXAMPLES=$EXAMPLES_DIR DECODES=$DECODES PY=$PY"
$PY scripts/eval_lora.py --framework "$FRAMEWORK" --adapter "$ADAPTER" --adapter-id "$ADAPTER_ID" --bits "$BITS" --max-seq-len "$MAX_SEQ_LEN" --granularity "$GRAN" --decodes "$DECODES" --examples-dir "$EXAMPLES_DIR" "${opt[@]}"
