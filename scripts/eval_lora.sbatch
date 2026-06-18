#!/bin/bash
#SBATCH --job-name=eval_lora
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
#SBATCH --qos=aisc                     # AISC QOS = faster scheduling
#SBATCH --gres=gpu:h100:2              # default; override per run: sbatch --gres=gpu:h100:N
#SBATCH --constraint=ARCH:X86          # uv is x86-only; ARM GPU nodes crash
#SBATCH --exclude=gx13v1               # flaky node
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_lora_%j.out
#SBATCH --error=logs/eval_lora_%j.err

# Generate data/test summaries for ONE adapter (all examples x decode presets).
# Env-driven so one wrapper serves PEFT / Unsloth / Keras. Submit via
# scripts/eval_submit.sh, which sets these and the per-framework --gres/--time.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

set -a
[ -f .env ] && . ./.env
set +a

: "${FRAMEWORK:?Set FRAMEWORK=peft|unsloth|keras}"
: "${ADAPTER:?Set ADAPTER=path/to/adapter}"
: "${ADAPTER_ID:?Set ADAPTER_ID=short_id}"
BASE_MODEL="${BASE_MODEL:-}"           # empty = read from adapter_config.json
BITS="${BITS:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
GRAN="${GRAN:-per-top}"
DECODES="${DECODES:-baseline,antirep}"
ONLY="${ONLY:-}"
PRESET="${PRESET:-}"
LORA_RANK="${LORA_RANK:-8}"
PY="${PY:-uv run python}"              # peft: 'uv run python'; alt: abs venv python

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/sc/projects/sci-aisc/pilotproject-automatic-protocols/hf-cache}"

# Unsloth's Triton kernels JIT-compile a helper that #includes Python.h.
if [ "$FRAMEWORK" = "unsloth" ]; then
  PYH="$(find "$HOME/.local/share/uv/python" -name Python.h -path '*python3.12*' 2>/dev/null | head -1)"
  [ -n "$PYH" ] && export CPATH="$(dirname "$PYH"):${CPATH:-}"
fi
[ "$FRAMEWORK" = "keras" ] && export KERAS_BACKEND=jax XLA_PYTHON_CLIENT_MEM_FRACTION=1.0

opt=()
[ -n "$BASE_MODEL" ] && opt+=(--base-model "$BASE_MODEL")
[ -n "$PRESET" ]     && opt+=(--preset "$PRESET")
[ -n "$ONLY" ]       && opt+=(--only "$ONLY")

echo "[$(date -Is)] FRAMEWORK=$FRAMEWORK ADAPTER=$ADAPTER ID=$ADAPTER_ID BITS=$BITS MAXSEQ=$MAX_SEQ_LEN DECODES=$DECODES PY=$PY"
$PY scripts/eval_lora.py \
    --framework  "$FRAMEWORK" \
    --adapter    "$ADAPTER" \
    --adapter-id "$ADAPTER_ID" \
    --bits       "$BITS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --granularity "$GRAN" \
    --decodes    "$DECODES" \
    --lora-rank  "$LORA_RANK" \
    "${opt[@]}"
