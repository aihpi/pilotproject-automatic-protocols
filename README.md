<div style="background-color: #ffffff; color: #000000; padding: 10px;">
<img src="00_aisc\img\logo_aisc_bmftr.jpg">
<h1> Your title.
</div>

Your Project Description with a nice image

## Features

- **Key Feature 1**: A description of the Key features
- **Key Feature 2**: A description of the Key features

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional, but recommended for faster performance)

### Quick Start

1. Clone the repository:
   ```bash
   git clone ...
   cd ...
   ```

2. Run the setup or install dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Access the application:
   - Frontend: ...
   - Backend API: ...

## User Guide

### Using the Tool
1. A brief description of using the tool.
2. Be clear and simple.

### Recommendations
Any additional hints for using the tool.


## Limitations

- **Limitation 1**: List of Limitations
- **Limitation 2**: List of Limitations


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

- [Reference 1](https://hpi.de/kisz)
- [Reference 2](https://hpi.de/kisz)

## Author
- [Your Name](https://hpi.de/kisz)

## License


---

## Acknowledgements
<img src="00_aisc/img/logo_bmftr_de.png" alt="drawing" style="width:170px;"/>

The [AI Service Centre Berlin Brandenburg](http://hpi.de/kisz) is funded by the [Federal Ministry of Research, Technology and Space](https://www.bmbf.de/) under the funding code 16IS22092.
