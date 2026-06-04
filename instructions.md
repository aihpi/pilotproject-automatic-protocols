# Transcript → smart summary pipeline (LoRA)

This pipeline fine-tunes an instruction LLM to turn committee **transcripts** into **smart summaries**. This is a style/structure-transfer task, well suited to parameter-efficient fine-tuning. The default base is **`google/gemma-4-E2B-it`** (small, ungated — for implementation and testing); scale up with `--base-model google/gemma-4-31B-it`. Training is **QLoRA (4-bit)** by default on a single 80 GB H100 (`aisc-batch`).

Approx VRAM for the 31B (seq 4096, batch 1, gradient checkpointing, AdamW-8bit, LoRA r=16): **QLoRA 4-bit ≈ 24–28 GB** (base ≈ 16 GB); **16-bit LoRA ≈ 68–72 GB** (base ≈ 62 GB, fits one H100 tightly). Full fine-tuning would need ~500 GB.

Gemma 4 is multimodal (text+vision+audio), but `AutoModelForCausalLM` loads the text path and training is text-only. Two Gemma-specific settings are baked into `train_lora.py`: attention defaults to **`sdpa`** (no flash-attn build needed), and `--exclude-modules` drops the **vision/audio towers** (their projections share the `q_proj`/`k_proj` names but use a wrapper layer PEFT can't adapt). Both are no-ops for plain text models, so the script stays general.


## A. Data conversion

All converters take `--input` (a file **or** a directory) and `--out-dir`, skip existing outputs unless `--overwrite`, and use exit codes 0/1/2 like the audio scripts.

Both converters use **Docling** (layout-aware markdown export, OCR disabled — protocols are born-digital). Docling downloads its layout/table models on first run (cached under `HF_HOME`) and is slow (~50 s per protocol PDF), so for large batches run it via SLURM / in parallel.

```bash
# protocol PDF → markdown (docling, layout + table aware)
uv run python scripts/pdf_to_markdown.py --input data/protocols/pdf --out-dir data/protocols/md

# strip everything before the first "Zu TOP 1" (header/attendance/agenda can't be
# inferred from the transcript, so it must not be a training target)
uv run python scripts/preprocess_protocol.py --input data/protocols/md --out-dir data/protocols/md_clean

# transcript DOCX → markdown, preserving <SD-TOP>/<SD-SPK> tags (already-clean data)
uv run python scripts/docx_to_markdown.py --input data/transcripts/doc --out-dir data/transcripts/md
```

> **Note**: Docling pulls OpenCV; the cluster has no `libGL`, so `pyproject.toml` pins `opencv-python-headless` and a `[tool.uv]` override drops the non-headless `opencv-python` on Linux. `torchvision` is also pinned to the cu128 index to match `torch`. Just `uv sync`; no system libraries needed.


## B. Build the dataset

`build_dataset.py` pairs transcripts to protocols by filename match, segments them and writes chat-format JSONL. By default it reads transcripts from `data/transcripts/md` and protocols from `data/protocols/md_clean`, cleans them and splits **per agenda item** (`per-top`) — shorter sequences and many more examples. Use `--granularity document` for one record per session.

```bash
uv run python scripts/build_dataset.py \
    --transcript-dir data/transcripts/md --protocol-dir data/protocols/md_clean \
    --granularity per-top --out-dir data/train
```

Output `data/train/{train,val}.jsonl`, one record per line:

```json
{"messages": [{"role": "system", "content": "Du bist Protokollführer/in …"},
              {"role": "user", "content": "<transcript / TOP segment>"},
              {"role": "assistant", "content": "<protocol / TOP section>"}],
 "meta": {"stem": "ail 11 sitzung", "top": 1, "strategy": "per-top", ...}}
```

The train/val split is **by session** (a session's items never straddle the split). The pairing log and a token-length summary print to stderr — eyeball them before training.

## C. Train the adapter (GPU / SLURM)

```bash
TRAIN_JSONL=data/train/train.jsonl sbatch scripts/train_lora.sbatch
```

| Variable      | Default                              |
|---------------|--------------------------------------|
| `TRAIN_JSONL` | (required)                           |
| `VAL_JSONL`   | `data/train/val.jsonl` (if present) |
| `OUT_DIR`     | `results/lora_adapter`                   |
| `BASE_MODEL`  | `google/gemma-4-E2B-it`              |
| `MAX_SEQ_LEN` | `4096`                               |
| `BITS`        | `4` (QLoRA)                          |
| `EPOCHS`      | `3`                                  |
| `MAX_STEPS`   | `-1` (use epochs; set small to smoke-test) |

Scale up to the 31B with `BASE_MODEL=google/gemma-4-31B-it`. `HF_HOME` defaults to shared project storage to keep downloads off the home quota.

**HuggingFace token**: `cp .env_example .env` and set `HF_TOKEN=…` (from https://huggingface.co/settings/tokens). The `train_lora.sbatch` / `infer_summary.sbatch` launchers source `.env` automatically — this clears the "unauthenticated requests" warning, lifts download rate limits, and unlocks gated repos. `.env` is gitignored; never commit it.

Key `train_lora.py` flags: `--bits {4,16}`, `--lora-r/--lora-alpha`, `--epochs`, `--max-steps`, `--attn {sdpa,flash_attention_2,eager}` (sdpa default), `--exclude-modules`.


## D. Inference (GPU / SLURM)

```bash
INPUT="data/transcripts/md/example_transcript.md" \
    ADAPTER=results/lora_adapter sbatch scripts/infer_summary.sbatch
```

`infer_summary.py` loads the base model (4-bit) + adapter, and with `--granularity per-top` (default) summarises each numbered `<SD-TOP>` segment and concatenates them under `Zu TOP N` headings. Tune with `--temperature`, `--top-p`, `--max-new-tokens`; `--stream` echoes tokens to stderr.


## E. flash-attn note

Training defaults to **`--attn sdpa`**, which needs no extra build, so flash-attn is optional and intentionally **not** in `pyproject.toml`. If you want it (a bit faster / less memory at long sequence lengths), install it once and pass `--attn flash_attention_2`:

```bash
uv pip install flash-attn --no-build-isolation
```
