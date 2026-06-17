# Alternative LoRA implementations (branch `fix/LoRA_alternative_implementations`)

Fresh, framework-diverse LoRA training scripts to beat the gemma-4-31B long-context
**OOM** (root cause: the `seq×vocab` logits tensor, vocab = 262144 — not the weights).
The original `scripts/train_lora.py` is left untouched. The fused-loss family (liger,
cut-cross-entropy) on `fix/OOM_issues` either OOMs or NaNs at long seq; these take
*different* angles. Each is minimal first — add complexity only when the minimal run fails.

All four read the same chat-`messages` JSONL (`scripts/build_dataset.py` output) and the
**pinned comparison dataset** `data/train_no_docs/{train,val}.jsonl` (cap 32768) so results
are comparable. Adapters land in `results/<framework>_lora/`; eval with `scripts/infer_summary.py`.

## The four approaches

| # | Framework | Angle on the OOM | GPUs | Files |
|---|---|---|---|---|
| 1 | **Unsloth** | single-GPU, fused CE + flash-attn, never materialises full logits | 1×H100 | `scripts/train_lora_unsloth.py` + `.sbatch` |
| 2 | **Keras + JAX** | different stack/autodiff → dodges the PyTorch CE NaN; ModelParallel to escalate | 1→N | `scripts/train_lora_keras.py` + `.sbatch` |
| 3 | **FSDP / ZeRO-3** | shard weights/grads/optimizer; frees GPU 0 (no `device_map=auto`) | 2–4×H100 | `scripts/train_lora_fsdp.py`, `configs/fsdp.yaml`, `configs/zero3.json`, `.sbatch` |
| 4 | **Axolotl** | config-only QLoRA+FSDP; context parallelism for the 162k records | 4×H100 | `axolotl/gemma4_qlora.yml` + `scripts/train_lora_axolotl.sbatch` |

Dependencies are isolated uv extras (`[project.optional-dependencies]`): `unsloth`, `keras`,
`fsdp`, `axolotl`. Each sbatch runs `uv run --extra <name>`. If an extra clashes with the base
torch stack, build it in a dedicated venv.

## How to run (smoke first, then target)

Cluster only — `sbatch` (never `srun` background / login node). All jobs already set
`--account=aisc --partition=aisc-batch --constraint=ARCH:X86`.

```bash
# 1) Unsloth — smoke on small model, then 31B @ 32768
BASE_MODEL=google/gemma-4-E2B-it MAX_SEQ_LEN=4096 MAX_STEPS=20 sbatch scripts/train_lora_unsloth.sbatch
sbatch scripts/train_lora_unsloth.sbatch                                   # 31B, cap 32768

# 2) Keras+JAX — smoke on a small preset, then scale + --model-parallel
PRESET=gemma3_instruct_1b MAX_SEQ_LEN=4096 LIMIT=20 sbatch scripts/train_lora_keras.sbatch

# 3) FSDP (or ZeRO-3) — smoke 2 GPUs, then 4
BASE_MODEL=google/gemma-4-E2B-it MAX_SEQ_LEN=4096 MAX_STEPS=20 NGPU=2 sbatch scripts/train_lora_fsdp.sbatch
BACKEND=zero3 sbatch scripts/train_lora_fsdp.sbatch                         # DeepSpeed variant

# 4) Axolotl — smoke via CLI overrides, then full
OVERRIDES="base_model=google/gemma-4-E2B-it sequence_len=4096 max_steps=20" sbatch scripts/train_lora_axolotl.sbatch
sbatch scripts/train_lora_axolotl.sbatch
```

Pass = finite `grad_norm` + loss ~1–4 + adapter written (watch `logs/train_*_<jid>.out`;
`grad_norm: nan` = divergence). Then push cap → `data/train_no_docs_cap65k`, `_uncapped` (162k).

## Open risks (resolve early)
- **Unsloth**: confirm gemma-4 is supported; may need an `unsloth/gemma-4-*-bnb-4bit` repo.
- **Keras**: a gemma-4-31B KerasHub preset may not exist/convert — if so, document + fall back
  to the largest available Gemma preset.
- **FSDP+4-bit**: minimal FSDP path is bf16 LoRA; QLoRA-under-FSDP needs extra wiring — use
  ZeRO-3 for the 4-bit route first.
- **Axolotl**: `fsdp_transformer_layer_cls_to_wrap` must match the real gemma-4 decoder class
  name (`Gemma4DecoderLayer` assumed — verify against the installed transformers).

## Comparison table (fill in after runs)

| framework | OOM solved? | max stable cap | precision | peak VRAM | GPUs | wall-clock/epoch | final eval_loss | notes |
|---|---|---|---|---|---|---|---|---|
| Unsloth   |  |  |  |  | 1 |  |  |  |
| Keras+JAX |  |  |  |  |  |  |  |  |
| FSDP      |  |  |  |  | 4 |  |  |  |
| ZeRO-3    |  |  |  |  | 4 |  |  |  |
| Axolotl   |  |  |  |  | 4 |  |  |  |
