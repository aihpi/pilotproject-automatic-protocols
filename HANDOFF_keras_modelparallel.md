# Handoff — Keras/JAX gemma-4-31B LoRA via ModelParallel (blocked on JAX/NCCL)

Date: 2026-06-18. Branch: `fix/LoRA_alternative_implementations`. Script:
`scripts/train_lora_keras.py`, venv `.venv-keras` (keras 3 / keras-hub 0.29.1 / jax[cuda12]).
Audience: whoever resumes the Keras track. **The model-parallel *logic* works; the blocker is a
cluster-level JAX↔NCCL collective failure.**

## Goal
Train a LoRA on `gemma-4-31B-it` in KerasHub. Single-GPU OOMs (62 GB bf16 weights), so 31B needs
**tensor/model parallelism** across GPUs (`keras.distribution.ModelParallel`).

## What already works (don't redo)
1. **Use `keras_hub.models.Gemma4CausalLM.from_preset(...)` explicitly.** The auto-dispatch
   `keras_hub.models.CausalLM.from_preset("hf://google/gemma-4-…-it")` resolves to
   `Gemma4AssistantCausalLM`, whose `__init__` raises *missing 4 required positional args*
   (`backbone_hidden_size, num_centroids, centroid_intermediate_top_k, use_ordered_embeddings`).
   `Gemma4CausalLM.from_preset` constructs fine. (Script already does this for gemma-4 presets.)
2. **gemma-4 tensor-parallel `layout_map` (verified against `Gemma4Backbone.weights`)** — model
   loads, **shards across N GPUs, and reaches `fit()`**. 2D mesh `(batch=1, model=N)`,
   `batch_dim_name="batch"`. Rules (shard on `"model"` axis):
   - `token_embedding/embeddings` → `("model", None)`  (vocab 262144)
   - `per_layer_token_embedding/embeddings` → `("model", None)`  (262144 × per-layer dim — huge)
   - `per_layer_model_projection/kernel` → `(None, "model")`
   - `decoder_block.*attention/query/kernel` → `("model", None, None)`  (shard query heads)
   - `decoder_block.*attention/attention_output/kernel` → `("model", None, None)`
   - `decoder_block.*ffw_gating.*/kernel` → `(None, "model")`;  `…/ffw_linear/kernel` → `("model", None)`
   - `decoder_block.*per_layer_input_gate/kernel` → `(None, "model")`;  `…/per_layer_up_proj/kernel` → `("model", None)`
   - **KV is GQA (1 head)** → leave `key`/`value` replicated (no rule); norms replicated; vision/
     audio encoders replicated (irrelevant to text-only LoRA).

## The blocker
During `gemma.fit(...)` (first step), on **both** 1- and 2-metric configs and with
`NCCL_P2P_DISABLE=1`:
```
jax.errors.JaxRuntimeError: INTERNAL: NCCL operation ncclAllReduce(...) failed:
invalid argument (run with NCCL_DEBUG=WARN for details). [executable_name='jit_greater']
```
`NCCL_DEBUG=WARN` emitted no extra detail. The `jit_greater` op is incidental (first collective).
This is a **cluster-specific JAX-distributed / NCCL runtime** problem, not the sharding map.

## Repro (cheap)
```
srun --account=aisc --qos=aisc --partition=aisc-shortrun --constraint=ARCH:X86 \
  --gres=gpu:h100:2 --ntasks=1 --mem=128G --time=00:45:00 --exclude=gx13v1 \
  .venv-keras/bin/python scripts/train_lora_keras.py \
    --preset hf://google/gemma-4-E2B-it --train-jsonl data/train_no_docs/train.jsonl \
    --max-seq-len 4096 --limit 20 --epochs 1 --model-parallel
```

## Fix directions to try
- `NCCL_DEBUG=INFO` (not WARN) to capture the real NCCL failure line.
- Check **jaxlib ↔ system NCCL version** match; try the NCCL bundled with jaxlib vs system; pin a
  known-good jax[cuda12] build.
- Env: `NCCL_SHM_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_NVLS_ENABLE=0`; `XLA_FLAGS` for collectives.
- Try `jax.distributed.initialize()` (even single-node) and/or `JAX_NUM_PROCESSES`/coordinator;
  confirm single-process-multi-device vs multi-process expectations on this Slurm setup.
- Test a *minimal* 2-GPU JAX all-reduce (no Keras) to isolate JAX/NCCL from KerasHub.

## Files
- `scripts/train_lora_keras.py` — `args.model_parallel` block (layout_map) + `Gemma4CausalLM`
  selection + `weighted_metrics=None` under model-parallel.
