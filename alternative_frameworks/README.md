# Alternative LoRA implementations

> **Reorganised 2026-06-22.** The **Unsloth** approach won and is now the project's
> **canonical trainer** at `scripts/train_lora_unsloth.py` (launcher
> `scripts/train_lora_unsloth.sbatch`) — it is the recipe formerly called
> `train_lora_unsloth_v2.py`. Everything in *this* folder is the parked, non-canonical
> set: the original PEFT trainer (`train_lora_PEFT.py`, was `scripts/train_lora.py`),
> the **deprecated** original Unsloth recipe (`train_lora_unsloth_deprecated.py`), plus
> FSDP, Keras and Axolotl. They are kept for reference and require SLURM/the HPI cluster.
> Shared helpers (`alt_utils.py`, `model_utils.py`, the eval harness) stayed in `scripts/`;
> the moved trainers add `scripts/` to `sys.path` to import them. Paths below are
> repo-root-relative (run `sbatch` from the repo root). The historical notes below
> predate the move.

Fresh, framework-diverse LoRA training scripts to beat the gemma-4-31B long-context
**OOM** (root cause: the `seq×vocab` logits tensor, vocab = 262144 — not the weights).
The original `scripts/train_lora.py` is left untouched. The fused-loss family (liger,
cut-cross-entropy) on `fix/OOM_issues` either OOMs or NaNs at long seq; these take
*different* angles. Each is minimal first — add complexity only when the minimal run fails.

All four read the same chat-`messages` JSONL (`scripts/build_dataset.py` output) and the
**pinned comparison dataset** `data/train_no_docs/{train,val}.jsonl` (cap 32768) so results
are comparable. Each run writes a timestamped folder `results/YYYYMMDD-HHMMSS/` (adapter/
weights + tokenizer + `train_log.md`; framework/model/dataset recorded in the log content, not
the folder name) — same convention as `scripts/train_lora.py`. Eval with `scripts/infer_summary.py`.

## The four approaches

| # | Framework | Angle on the OOM | GPUs | Files |
|---|---|---|---|---|
| 1 | **Unsloth** (deprecated recipe) | single-GPU, fused CE + flash-attn, never materialises full logits | 1×H100 | `alternative_frameworks/train_lora_unsloth_deprecated.py` (no maintained launcher; canonical recipe is `scripts/train_lora_unsloth.py`) |
| 2 | **Keras + JAX** | different stack/autodiff → dodges the PyTorch CE NaN; ModelParallel to escalate | 1→N | `alternative_frameworks/train_lora_keras.{py,sbatch}` |
| 3 | **FSDP / ZeRO-3** | shard weights/grads/optimizer; frees GPU 0 (no `device_map=auto`) | 2–4×H100 | `alternative_frameworks/train_lora_fsdp.py`, `alternative_frameworks/configs/fsdp.yaml`, `alternative_frameworks/configs/zero3.json`, `.sbatch` |
| 4 | **Axolotl** | config-only QLoRA+FSDP; context parallelism for the 162k records | 4×H100 | `alternative_frameworks/axolotl/gemma4_qlora.yml` + `alternative_frameworks/train_lora_axolotl.sbatch` |

### Environment — standalone venvs (NOT uv extras)

The original plan put each framework in a `pyproject` extra; that **does not work** — `uv` always
co-resolves the base deps (`torch>=2.11`, `numpy>=2.4.4`, `liger-kernel`) with the extra, and
Axolotl/Keras pin older/incompatible torch+numpy → unsatisfiable. So each framework gets its own
**standalone venv** built outside the project resolver. The `pyproject` extras are kept only as a
dependency record. Build once on a **run node** (`rx01`/`rx02` — login node home is `noexec`, so uv
can't execute there; run nodes can, and have internet):

```bash
cd <worktree>
uv venv .venv-unsloth --python 3.12 && uv pip install --python .venv-unsloth unsloth trl peft datasets bitsandbytes tensorboard
uv venv .venv-keras   --python 3.12 && uv pip install --python .venv-keras   keras keras-hub "jax[cuda12]"
uv venv .venv-axolotl --python 3.12 && uv pip install --python .venv-axolotl axolotl
# FSDP reuses the MAIN checkout's base .venv (transformers/peft/trl/accelerate) — no build.
```

### How to run (smoke first, then target)

Train via **`sbatch`** for real runs; for a quick smoke a detached **`srun` on `aisc-shortrun`**
starts in seconds (the `aisc-batch` queue can be ~2 days for HPI jobs). All sbatch already set
`--account=aisc --qos=aisc --partition=aisc-batch --constraint=ARCH:X86 --exclude=gx13v1`.

```bash
# 1) Unsloth is now CANONICAL — use scripts/train_lora_unsloth.sbatch (see root README).
#    The deprecated original recipe (alternative_frameworks/train_lora_unsloth_deprecated.py)
#    has no maintained launcher.

# 2) Keras+JAX — smoke on a small preset, then scale + MODEL_PARALLEL=1
PRESET=gemma3_instruct_1b MAX_SEQ_LEN=4096 LIMIT=20 sbatch alternative_frameworks/train_lora_keras.sbatch

# 3) FSDP (or ZeRO-3) — smoke 2 GPUs, then 4
BASE_MODEL=google/gemma-4-E2B-it MAX_SEQ_LEN=4096 MAX_STEPS=20 NGPU=2 sbatch alternative_frameworks/train_lora_fsdp.sbatch
BACKEND=zero3 sbatch alternative_frameworks/train_lora_fsdp.sbatch         # DeepSpeed variant

# 4) Axolotl — smoke via CLI overrides, then full
OVERRIDES="base_model=google/gemma-4-E2B-it sequence_len=4096 max_steps=20" sbatch alternative_frameworks/train_lora_axolotl.sbatch
sbatch alternative_frameworks/train_lora_axolotl.sbatch
```

Offline-cache note: pass a **local snapshot path** as `BASE_MODEL` rather than the HF id —
`HF_HUB_OFFLINE=1` makes Unsloth's config lookup fail, and compute nodes may lack internet.
Snapshot: `$HF_HOME/hub/models--google--gemma-4-31B-it/snapshots/<hash>/`.

Pass = finite `grad_norm` + sane loss + adapter written (watch `logs/train_*_<jid>.out`;
`grad_norm: nan` = divergence). Then push cap → `data/train_no_docs_cap65k`, `_uncapped` (162k).

## Open risks (resolve early)
- **Unsloth**: confirm gemma-4 is supported; may need an `unsloth/gemma-4-*-bnb-4bit` repo.
- **Keras**: a gemma-4-31B KerasHub preset may not exist/convert — if so, document + fall back
  to the largest available Gemma preset.
- **FSDP+4-bit**: minimal FSDP path is bf16 LoRA; QLoRA-under-FSDP needs extra wiring — use
  ZeRO-3 for the 4-bit route first.
- **Axolotl**: `fsdp_transformer_layer_cls_to_wrap` must match the real gemma-4 decoder class
  name (`Gemma4DecoderLayer` assumed — verify against the installed transformers).

## Live status (2026-06-17)

- **Unsloth ✓ FULLY VALIDATED**: E2B@4096 smoke + **31B-it @ 32768 full 3-epoch run on a SINGLE
  H100** — no OOM, final train loss 0.115, **eval_loss 1.289→1.098→1.055**. Adapter:
  `results/20260617-182728/`. eval ≈ the CCE baseline (~1.0) → comparable results, on one GPU.
- **Keras ✓**: gemma-4 supported (`Gemma4Backbone`/`convert_gemma4`, auto-dispatch
  `CausalLM.from_preset`); smoke trained + saved (gemma-3-1b placeholder). 31B path = same script
  + `MODEL_PARALLEL=1`.
- **FSDP ✓**: E2B smoke trains + saves (FSDP1; FSDP2 broke PEFT LoRA). Save = `FULL_STATE_DICT`.
- **Axolotl ✗ deprioritized**: cleared 6 issues (optimizer, torchvision, cu128 driver, wrap-class
  `Gemma4TextDecoderLayer`, eager-attn for head_dim>256, CCE) but hits a persistent 32 GB OOM at
  E2B/4096/2-GPU under FSDP1/2 — a transient full-precision load neither FSDP version shards. It's
  a config-wrapper over the FSDP track (#3) already validated, so low marginal value.

Per-framework gemma-4 gotchas (all resolved except Axolotl): pass a **local snapshot path**
(offline); chat template needs **typed-parts content + system folded into user**; Unsloth needs a
batch-aware `formatting_func`; Keras -it preprocessor wants **{prompts,responses}** + TF off-GPU;
FSDP must **pre-render text** (no `{% generation %}`) and use **FSDP1 + FULL_STATE_DICT**.

## Results structure

Every run is `results/YYYYMMDD-HHMMSS/` matching `scripts/train_lora.py`:
- Unsloth / FSDP — PEFT adapter (`adapter_config.json`, `adapter_model.safetensors`), tokenizer,
  `train_log.md`, `checkpoint-*/`, `runs/` (TensorBoard). Directly loadable by `infer_summary.py`.
- Axolotl — its own run dir (adapter + resolved config + checkpoints) at the same stamped path.
- **Keras — exception**: KerasHub LoRA saves a single `lora.weights.h5` (+ `train_log.md`), not a
  PEFT adapter. It needs a Keras-side loader for inference, not `infer_summary.py`. Flagged so the
  comparison harness treats Keras separately.

## Comparison vs the CCE baseline (`fix/OOM_issues`)

Baseline 31B adapters to compare against (consistency / VRAM / runtime / quality):
`results/cce_nodocs_cap32k` (4-bit@32768, eval ~1.0, best), `results/smoke_cce_sb`
(bf16@4096, eval 1.54), `results/smoke_qlora4k` (4-bit@4096, eval 1.37).

Method: once an alt approach has a 31B adapter trained on `data/train_no_docs`, run
`scripts/infer_summary.py` (base 31B + adapter) on a fixed held-out transcript and diff the
generated protocols against the CCE baseline; log eval_loss + peak VRAM + wall-clock.

| framework | OOM solved? | max cap reached | precision | GPUs | eval_loss | notes |
|---|---|---|---|---|---|---|
| **CCE (baseline)** | yes | 32768 | 4-bit | multi | ~0.98 | `fix/OOM_issues`; `results/20260617-172746` |
| **Unsloth** | **yes** | **32768** | 4-bit | **1** | **1.055** | `results/20260617-182728`; eval ≈ CCE on ONE GPU. Uncapped(162k) OOMs even here. Adapter needs Unsloth runtime to load. |
| **FSDP + CCE** | **yes** | **32768** | bf16 | 4 | (3.96 train, short) | `results/20260618-103722`; needed CCE loss (logits) + CPATH. **Uncapped OOMs** (attention term, not logits). Stock-PEFT adapter. |
| Keras+JAX | smoke ✓ | 4096 (smoke) | bf16 | 1→N | — | gemma-4 loads (`Gemma4Backbone`); 31B needs ModelParallel layout_map; non-PEFT `.h5`. |
| Axolotl | ✗ | — | 4-bit | 2–4 | — | persistent 32 GB OOM; deprioritized (wrapper over FSDP track). |

### Caps: 32768 solved; 65536 in progress; 162k = future work
- **32768** — solved (Unsloth 1-GPU, FSDP+CCE 4-GPU, CCE baseline).
- **65536** — Unsloth reaches it on 1 H100 (run in progress); FSDP+CCE needs 8 GPU (4 OOMs).
- **Uncapped 162k** — **accepted as out of reach for now**. Both Unsloth (1 H100) and FSDP+CCE
  (4–8 H100) OOM: CCE removes the `seq×vocab` logits term, but **attention/activation at 162k**
  still blows up (gemma-4 head_dim>256 also blocks flash-attn). True 162k needs **sequence/
  context parallelism** (Axolotl `context_parallel` / FSDP ring-attention) — future work, per
  handoff §6 B. 65536 covers ~all real sessions.

### Keras 31B — model-parallel works, blocked on JAX/NCCL (parked)
Sharding logic is done and **accepted** (model shards across GPUs, reaches `fit()`): use
`Gemma4CausalLM.from_preset` (auto `CausalLM` mis-picks a broken Assistant class) + a verified
gemma-4 tensor-parallel `layout_map`. Blocked on a cluster-level `ncclAllReduce` "invalid
argument" during `fit` — a JAX-distributed/NCCL infra issue, not the sharding. Parked; full repro
+ fix directions in `tmp/HANDOFF_keras_modelparallel.md`.

### Protocol quality (held-out AIL_6 / AIK_8_1 / HA_8_4, per-top) — see `tmp/lora_cmp/`
Degradation (raw-transcript echo + repetition) is **dominated by the inference script, not the
adapter**: protocols via `infer_summary.py` (proper 2-pass `split_transcript_by_top`) are clean
(CCE, FSDP: ~0 raw-timestamp leaks); via `infer_unsloth.py` (crude single-pass split, no
repetition penalty) they degrade badly (100s of leaked lines). Fix tracked in
`tmp/HANDOFF_repetition_fix.md` (separate branch).
