# Transcript → smart summary pipeline (LoRA)

This pipeline fine-tunes an instruction LLM to turn committee **transcripts** into **smart summaries** (Ausschussprotokolle). This is a style/structure-transfer task, well suited to parameter-efficient fine-tuning. The default base is **`google/gemma-4-E2B-it`** (small, ungated — for implementation and testing); scale up with `--base-model google/gemma-4-31B-it`. Training is **QLoRA (4-bit)** by default on a single 80 GB H100 (`aisc-batch`).

Approx VRAM for the 31B (seq 4096, batch 1, gradient checkpointing, AdamW-8bit, LoRA r=16): **QLoRA 4-bit ≈ 24–28 GB** (base ≈ 16 GB); **16-bit LoRA ≈ 68–72 GB** (base ≈ 62 GB, fits one H100 tightly). Full fine-tuning would need ~500 GB.

Gemma 4 is multimodal (text+vision+audio), but `AutoModelForCausalLM` loads the text path and training is text-only. Two Gemma-specific settings are baked into `train_lora.py`: attention defaults to **`sdpa`** (no flash-attn build needed), and `--exclude-modules` drops the **vision/audio towers** (their projections share the `q_proj`/`k_proj` names but use a wrapper layer PEFT can't adapt). Both are no-ops for plain text models, so the script stays general.

All paths below assume the corpus lives under **`data/`** (gitignored). Stages run in order; each writes a new sub-folder so you can inspect intermediates.

```
data/protocols/pdf/                      raw protocol PDFs (input)
data/protocols/md/                       A → markdown protocols
data/protocols/md_clean/   (+ cover/)    A → cleaned bodies + separated cover pages
data/transcripts/{wav,md}/               A → diarised transcripts (<SD-SPK>SPEAKER_NN, timestamps)
data/transcripts/md_top/   (+ top_reports/)   B → transcripts with <SD-TOP> agenda tags
data/transcripts/md_prepared/            B → speaker labels replaced with real names  ← dataset source
data/speaker_maps/                       B → per-session resolution reports
data/exclusions.json                     B → TOPs to drop (unresolved speakers)
data/train/{train,val}.jsonl             C → the LoRA dataset
results/lora_adapter/                    D → trained adapter
results/summaries/                       E → generated protocols
```

## Secrets (`.env`)

`cp .env_example .env` and fill in:
- `HF_TOKEN` — HuggingFace token (gated models, download rate limits; required for the diarisation model in transcription and for gated base models).
- `OPENAI_API_KEY` + `OPENAI_API_BASE` — the self-hosted gpt-oss endpoint used by stage **B** (TOP detection + speaker resolution). Example base: `https://api.aisc.hpi.de/`.

The SLURM launchers and the stage-B scripts source `.env` automatically. `.env` is gitignored; never commit it.


## A. Data conversion & cleaning

All converters take `--input` (a file **or** a directory) and `--out-dir`, skip existing outputs unless `--overwrite`, and use exit codes 0/1/2.

```bash
# 1) protocol PDF → markdown (Docling, layout + table aware; OCR off, born-digital)
uv run python scripts/pdf_to_markdown.py --input data/protocols/pdf --out-dir data/protocols/md

# 2) clean protocols: split off the cover (title/attendance/agenda) into md_clean/cover/,
#    drop the Anlagen section, page-footer tables, <!-- image -->, hyperlinks and attachment
#    footnotes. Verifies the cover boundary per file (OK/WARN/FAIL).
uv run python scripts/preprocess_protocol.py --input data/protocols/md --out-dir data/protocols/md_clean

# 3) transcripts: audio → diarised markdown (Whisper + pyannote) with <SD-SPK>SPEAKER_NN tags
#    and [HH:MM:SS --> HH:MM:SS] segments (NO agenda tags yet). Runs on GPU/SLURM:
INPUT_LIST=data/transcripts/manifest.txt OUT_DIR=data/transcripts/md DIARIZE=1 \
    sbatch scripts/transcribe.sbatch
#    (alternative: pre-existing transcript DOCX → markdown)
# uv run python scripts/docx_to_markdown.py --input data/transcripts/docx --out-dir data/transcripts/md
```

> **Note**: Docling pulls OpenCV; the cluster has no `libGL`, so `pyproject.toml` pins `opencv-python-headless` and a `[tool.uv]` override drops the non-headless `opencv-python` on Linux. `torchvision`/`torch`/`torchaudio` are pinned to the cu128 index. Just `uv sync`; no system libraries needed.

`preprocess_protocol.py` is also importable: `clean_protocol` and `split_cover` are reused by the stages below, so stages B and C clean protocols on the fly — passing them raw `data/protocols/md` works too.


## B. Prepare transcripts: agenda tags + real speaker names (LLM)

The diarised transcripts have **no `<SD-TOP>` agenda markers** and only generic `SPEAKER_NN` labels. Two scripts fix that, both **LLM-by-default** (gpt-oss-120b via `OPENAI_API_BASE`), processing sessions in parallel. Shared helpers live in `scripts/speaker_utils.py` (name/cover parsing) and `scripts/llm_utils.py` (client, JSON chat, parallel map). Pass `--no-llm` to fall back to regex/heuristics (no key needed).

```bash
# B1) tag agenda items: read the cover agenda, locate where the chair takes up each TOP
#     in the transcript, insert <SD-TOP>TOP N</SD>. Writes a per-session recall report.
uv run python scripts/tag_transcript_tops.py \
    --transcript-dir data/transcripts/md --protocol-dir data/protocols/md \
    --out-dir data/transcripts/md_top

# B2) resolve speakers: map each SPEAKER_NN to a real "Name (Rolle)" — committee members
#     from the cover attendance list, ministers/guests from the protocol prose, chair from
#     the cover "Vorsitz:" line, the rest by LLM. Substitutes names into the transcript and
#     writes an exclusions manifest of TOPs that still contain an unidentified speaker.
uv run python scripts/match_speakers.py \
    --transcript-dir data/transcripts/md_top --protocol-dir data/protocols/md \
    --out-transcript-dir data/transcripts/md_prepared \
    --report-dir data/speaker_maps --exclusions-out data/exclusions.json
```

`data/transcripts/md_prepared/` are the **final transcripts** (with `<SD-TOP>` tags and full names; any unresolved speaker stays `SPEAKER_NN`). Useful flags (both scripts): `--concurrency N` (parallel sessions), `--max-tokens` (completion budget incl. reasoning — raise if you see a *truncated/empty* warning), `--llm-model`, `--llm-base-url`, `--no-llm`. For `match_speakers.py`: `--max-unresolved-sentences N` excludes a TOP only if an unidentified speaker exceeds N sentences (default 2); `--content-threshold`.

Inspect `data/speaker_maps/*.json` (per-label name + method + conflicts) and `data/transcripts/md_top/top_reports/*.json` (cover vs. found TOPs) before building the dataset.


## C. Build the dataset

`build_dataset.py` pairs prepared transcripts to protocols by filename, splits **per agenda item** (`per-top`, default — shorter sequences, many more examples; `--granularity document` for one record per session), drops `--exclusions` TOPs, and writes chat-format JSONL. It cleans the protocol internally, so point `--protocol-dir` at raw `data/protocols/md` (or `md_clean`).

```bash
uv run python scripts/build_dataset.py \
    --transcript-dir data/transcripts/md_prepared --protocol-dir data/protocols/md \
    --exclusions data/exclusions.json --granularity per-top --out-dir data/train
```

Output `data/train/{train,val}.jsonl`, one record per line:

```json
{"messages": [{"role": "system",    "content": "Du bist Protokollführer/in …"},
              {"role": "user",       "content": "<SD-TOP>TOP 2</SD>\n<SD-SPK>Daniel Keller (SPD)</SD>\n[00:20:…] …"},
              {"role": "assistant",  "content": "Zu TOP 2:\n\nDer Hauptausschuss beschließt einstimmig (9 : 0 : 0) …"}],
 "meta": {"stem": "ha_1_", "top": 2, "strategy": "per-top", "src_tokens": 1308, "tgt_tokens": 255}}
```

The train/val split is **by session** (a session's items never straddle the split). The pairing log, exclusion count and a token-length summary print to stderr — eyeball them before training. Override the system prompt with `--system-prompt-file`; change the record schema in `make_record` if you need TRL's prompt/completion or plain-text formats instead of `messages`.


## D. Train the adapter (GPU / SLURM)

```bash
TRAIN_JSONL=data/train/train.jsonl sbatch scripts/train_lora.sbatch
```

| Variable      | Default                              |
|---------------|--------------------------------------|
| `TRAIN_JSONL` | (required)                           |
| `VAL_JSONL`   | `data/train/val.jsonl` (if present)  |
| `OUT_DIR`     | `results/lora_adapter`               |
| `BASE_MODEL`  | `google/gemma-4-E2B-it`              |
| `MAX_SEQ_LEN` | `4096`                               |
| `BITS`        | `4` (QLoRA)                          |
| `EPOCHS`      | `3`                                  |
| `MAX_STEPS`   | `-1` (use epochs; set small to smoke-test) |

Scale up to the 31B with `BASE_MODEL=google/gemma-4-31B-it`. `HF_HOME` defaults to shared project storage to keep downloads off the home quota.

Key `train_lora.py` flags: `--bits {4,16}`, `--lora-r/--lora-alpha/--lora-dropout`, `--target-modules`/`--exclude-modules`, `--epochs`, `--max-steps`, `--lr`, `--batch-size`/`--grad-accum`, `--max-seq-len`, `--packing`, `--attn {sdpa,flash_attention_2,eager}`.

> **Watch the sequence length:** per-TOP records with timestamps can exceed `MAX_SEQ_LEN=4096` real tokens and get truncated (losing part of the target). Raise `--max-seq-len` (e.g. 8192) for long sessions.


## E. Inference (GPU / SLURM)

```bash
INPUT="data/transcripts/md_prepared/example_Transkript.md" \
    ADAPTER=results/lora_adapter sbatch scripts/infer_summary.sbatch
```

`infer_summary.py` loads the base model (4-bit) + adapter, and with `--granularity per-top` (default) summarises each numbered `<SD-TOP>` segment and concatenates them under `Zu TOP N` headings. Feed it prepared transcripts so the input matches training (same system prompt, `<SD-TOP>` tags, `Name (Rolle)` speakers). Tune with `--temperature`, `--top-p`, `--max-new-tokens`; `--stream` echoes tokens to stderr.


## F. flash-attn note

Training defaults to **`--attn sdpa`**, which needs no extra build, so flash-attn is optional and intentionally **not** in `pyproject.toml`. If you want it (a bit faster / less memory at long sequence lengths), install it once and pass `--attn flash_attention_2`:

```bash
uv pip install flash-attn --no-build-isolation
```
