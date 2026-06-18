#!/bin/bash
# Submit data/test summarisation jobs for the LoRA adapters in the registry.
# One SLURM job per adapter (loads model once, all examples x decode presets).
#
#   scripts/eval_submit.sh                 # fan out ALL registry rows
#   scripts/eval_submit.sh cce_bf16        # only rows whose id matches 'cce_bf16'
#   SMOKE=1 scripts/eval_submit.sh         # one quick smoke job (short example, baseline)
#   DRYRUN=1 scripts/eval_submit.sh        # print the sbatch commands, submit nothing
#
# Never cancels/requeues other jobs; these queue with qos=aisc alongside training.
set -euo pipefail
cd "$(dirname "$0")/.."

REG="${REG:-tmp/lora_eval/registry.tsv}"
FILTER="${1:-}"
TIME="${TIME:-08:00:00}"
PARTITION="${PARTITION:-}"        # empty = sbatch default (aisc-batch)
mkdir -p logs tmp/lora_eval

if [ "${SMOKE:-0}" = "1" ]; then
  : "${SMOKE_ID:=cce_nodocs_cap32k}"
  FILTER="$SMOKE_ID"
  export ONLY="${ONLY:-short_ARD_1}"
  export DECODES_OVERRIDE="${DECODES_OVERRIDE:-baseline}"
  GPUS_OVERRIDE="${GPUS_OVERRIDE:-1}"
  PARTITION="${PARTITION:-aisc-interactive}"
  TIME="${TIME_OVERRIDE:-02:00:00}"
  echo "SMOKE: id~$FILTER only=$ONLY decodes=$DECODES_OVERRIDE gpus=$GPUS_OVERRIDE part=$PARTITION"
fi

submitted=0
while IFS=$'\t' read -r id fw adapter base bits maxseq gpus py preset rank decodes; do
  [ -z "${id:-}" ] && continue
  case "$id" in \#*) continue;; esac
  [ -n "$FILTER" ] && [[ "$id" != *"$FILTER"* ]] && continue

  # '-' placeholders in the TSV mean "empty" (tab-IFS collapses real empties)
  [ "$base" = "-" ] && base=""
  [ "$preset" = "-" ] && preset=""
  gpus="${GPUS_OVERRIDE:-$gpus}"
  [ -n "${DECODES_OVERRIDE:-}" ] && decodes="$DECODES_OVERRIDE"

  export FRAMEWORK="$fw" ADAPTER="$adapter" ADAPTER_ID="$id" BASE_MODEL="$base" \
         BITS="$bits" MAX_SEQ_LEN="$maxseq" DECODES="$decodes" PY="$py" \
         PRESET="$preset" LORA_RANK="$rank" ONLY="${ONLY:-}"

  args=(--gres="gpu:h100:$gpus" --time="$TIME" --job-name="eval_${id}" --export=ALL)
  [ -n "$PARTITION" ] && args+=(--partition="$PARTITION")

  echo "submit eval_${id}: fw=$fw gpus=$gpus bits=$bits maxseq=$maxseq decodes=$decodes"
  if [ "${DRYRUN:-0}" = "1" ]; then
    echo "  DRYRUN sbatch ${args[*]} scripts/eval_lora.sbatch"
  else
    sbatch "${args[@]}" scripts/eval_lora.sbatch
  fi
  submitted=$((submitted+1))
done < "$REG"
echo "done: $submitted job(s) ${DRYRUN:+(dry run)}"
