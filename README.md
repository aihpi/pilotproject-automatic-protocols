<div style="background-color: #ffffff; color: #000000; padding: 10px;">
<img src="00_aisc\img\logo_aisc_bmftr.jpg">
<h1> Automatic Protocols — committee transcripts → protocols
</div>

Fine-tune and run a LoRA adapter on `google/gemma-4` to turn Landtag/committee meeting
**transcripts** into structured German **protocols** (per agenda item / *Tagesordnungspunkt*).
The pipeline ingests raw audio/PDF/DOCX, prepares speaker-labelled and TOP-tagged transcripts,
builds an instruction dataset, trains a (Q)LoRA adapter, and generates protocol summaries.

## Pipeline

| stage | script | what it does |
|---|---|---|
| **A. Ingest** | `scripts/ingest_corpus.py`, `pdf_to_markdown.py`, `docx_to_markdown.py`, `transcribe.py` | stage the raw corpus; PDF/DOCX/audio → markdown |
| **B. Prepare** | `tag_transcript_tops.py`, `match_speakers.py` | tag agenda-item (TOP) boundaries and resolve speaker names → `data/transcripts/md_prepared/` |
| **C. Dataset** | `scripts/build_dataset.py` | pair transcripts↔protocols per TOP, write chat-format `train/val.jsonl` |
| **D. Train** | `scripts/train_lora.py` (+ `train_lora.sbatch`) | (Q)LoRA fine-tune on SLURM; long-context via cut-cross-entropy (`--cce`) |
| **E. Infer** | `scripts/infer_summary.py` (+ `infer_summary.sbatch`) | generate protocol summaries from new transcripts |

## Long-context training (the OOM fix)

gemma-4-31B has a ~262k-token vocabulary, so the standard loss materialises a
`seq_len × vocab` logits tensor that OOMs a single 80 GB H100 beyond ~32k tokens. Training
with **`--cce` (`USE_CCE=1`)** uses **cut-cross-entropy** to compute the loss without ever
building that tensor — enabling long and even uncapped sequences, numerically stable in bf16,
honouring Gemma's logit softcap. See `scripts/gemma4_cce_patch.py`.

## Quick start

```bash
# Train (SLURM) — long-context QLoRA on the 31B
USE_CCE=1 BITS=4 MAX_SEQ_LEN=32768 BASE_MODEL=google/gemma-4-31B-it \
  TRAIN_JSONL=data/train_no_docs/train.jsonl VAL_JSONL=data/train_no_docs/val.jsonl \
  sbatch --qos=aisc --gres=gpu:h100:2 --constraint=ARCH:X86 scripts/train_lora.sbatch

# Infer with a trained adapter
INPUT=data/transcripts/md_prepared/example_Transkript.md \
  ADAPTER=results/<YYYYMMDD-HHMMSS> sbatch scripts/infer_summary.sbatch
```

Each training run writes a self-contained `results/YYYYMMDD-HHMMSS/` folder (adapter + tokenizer
+ `train_log.md` + `README.md`). **Full pipeline details, flags, and SLURM/cluster notes are in
[`instructions.md`](instructions.md).**

## Limitations

- Built for German committee/Landtag protocols; the TOP-tagging heuristics are committee-oriented.
- Very long whole-document records (100k+ tokens) stress attention/activation memory even with
  cut-cross-entropy; cap or shard as needed.

## gemma-4-31B LoRA — alternative training stacks (status & open problems)

Branch `fix/LoRA_alternative_implementations` adds framework-diverse LoRA trainers to beat the
gemma-4-31B long-context OOM (root cause: the `seq×vocab`=262144 logits tensor). Full comparison
and run instructions: [`LORA_ALTERNATIVES.md`](LORA_ALTERNATIVES.md).

### Verified working (mergeable)
- **Unsloth, single H100** — `scripts/train_lora_unsloth.{py,sbatch}`. gemma-4-31B 4-bit @ **32768**,
  eval_loss **1.055** (≈ the CCE baseline ~0.98), no OOM. Reaches **65536** on one GPU (run confirmed
  in progress). Fused CE + offload. *Caveat:* its adapter only loads through Unsloth
  (`Gemma4ClippableLinear`) — inference via `scripts/infer_unsloth.py`, not stock PEFT.
- **FSDP + cut-cross-entropy, 4×H100** — `scripts/train_lora_fsdp.{py,sbatch}`, `configs/fsdp.yaml`,
  `scripts/gemma4_cce_patch.py`. gemma-4-31B bf16 @ **32768**, no OOM. FSDP shards the weights;
  **CCE is required** for the logits term (FSDP alone OOMs at 32768), plus `CPATH` for the Triton
  kernel. Stock-PEFT adapter (loads with `infer_summary.py`).

### Not working / needs further investigation (kept on this branch)
1. **Keras/JAX 31B model-parallel** — `scripts/train_lora_keras.{py,sbatch}`. The sharding *logic
   works*: `Gemma4CausalLM.from_preset` (the auto `CausalLM` mis-picks a non-constructable
   `Gemma4AssistantCausalLM`) + a verified gemma-4 tensor-parallel `layout_map`; the model shards
   across GPUs and reaches `fit()`. **Blocker:** `jax.errors.JaxRuntimeError: NCCL ncclAllReduce …
   invalid argument` (`jit_greater`) during the first step — persists without the accuracy metric
   and with `NCCL_P2P_DISABLE=1`. **Probable reason:** cluster-level JAX-distributed/NCCL
   mismatch (jaxlib↔NCCL version, topology, XLA flags) — not the sharding map. **Next steps:**
   `NCCL_DEBUG=INFO` for the real failure; verify jaxlib/NCCL versions; try
   `jax.distributed.initialize()`, `NCCL_SHM_DISABLE/IB_DISABLE/NVLS_ENABLE`; isolate with a
   minimal 2-GPU JAX all-reduce. Full detail: `tmp/HANDOFF_keras_modelparallel.md`.
2. **Axolotl** — `axolotl/gemma4_qlora.yml`, `scripts/train_lora_axolotl.sbatch`. Cleared 6
   issues (optimizer, torchvision, cu128 driver, `Gemma4TextDecoderLayer` wrap-class, eager-attn
   for head_dim>256, CCE) but hits a **persistent 32 GB transient OOM at E2B/4096/2-GPU** under
   both FSDP1 and FSDP2. **Probable reason:** a full-precision transient load neither FSDP version
   shards. **Next steps:** DeepSpeed ZeRO-3 backend instead of FSDP; disable axolotl preprocessing/
   packing; trace the 32 GB allocation. **Low priority** — it wraps the FSDP track that already works.
3. **Uncapped (162k-token records)** — OOMs on every stack (Unsloth 1-GPU; FSDP+CCE 4–8-GPU).
   **Probable reason:** CCE removes the logits term, but **attention/activation memory at 162k**
   still exceeds VRAM, and gemma-4 `head_dim>256` blocks FlashAttention. **Next steps:**
   **sequence/context parallelism** (Axolotl `context_parallel_size`, or FSDP + ring-attention) —
   the only family that scales the sequence dim. **65536 is the practical max reached.**
4. **Protocol repetition/echo at inference** — per-top summaries degrade from ~TOP 3 (echo raw
   transcript + repetition). **Probable reason:** crude `split_by_top` in `infer_unsloth.py` (vs
   the 2-pass `split_transcript_by_top`) + no `repetition_penalty`. Shown to be an *inference-script*
   issue, not adapter quality. **Next steps + evidence:** `tmp/HANDOFF_repetition_fix.md`.

## References

- [AI Service Centre Berlin-Brandenburg](https://hpi.de/kisz)

---

## Acknowledgements
<img src="00_aisc/img/logo_bmftr_de.png" alt="drawing" style="width:170px;"/>

The [AI Service Centre Berlin Brandenburg](http://hpi.de/kisz) is funded by the [Federal Ministry of Research, Technology and Space](https://www.bmbf.de/) under the funding code 16IS22092.
