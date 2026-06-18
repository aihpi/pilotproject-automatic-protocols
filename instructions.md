# Transcript → smart summary pipeline (LoRA)

This pipeline fine-tunes an instruction LLM to turn committee **transcripts** into **smart summaries** (Ausschussprotokolle). This is a style/structure-transfer task, well suited to parameter-efficient fine-tuning. The default base is **`google/gemma-4-E2B-it`** (small, ungated — for implementation and testing); scale up with `--base-model google/gemma-4-31B-it`. Training is **QLoRA (4-bit)** by default on a single 80 GB H100 (`aisc-batch`).

Approx VRAM for the 31B (seq 4096, batch 1, gradient checkpointing, AdamW-8bit, LoRA r=16): **QLoRA 4-bit ≈ 24–28 GB** (base ≈ 16 GB); **16-bit LoRA ≈ 68–72 GB** (base ≈ 62 GB, fits one H100 tightly). Full fine-tuning would need ~500 GB.

Gemma 4 is multimodal (text+vision+audio), but `AutoModelForCausalLM` loads the text path and training is text-only. Two Gemma-specific settings are baked into `train_lora.py`: attention defaults to **`sdpa`** (no flash-attn build needed), and `--exclude-modules` drops the **vision/audio towers** (their projections share the `q_proj`/`k_proj` names but use a wrapper layer PEFT can't adapt). Both are no-ops for plain text models, so the script stays general.

All paths below assume the corpus lives under **`data/`** (gitignored, a real working dir). The raw delivery is read-only shared storage, symlinked in as **`data/raw`**; stage **A0** stages it into the flat layout the rest of the pipeline expects. Stages run in order; each writes a new sub-folder so you can inspect intermediates.

```
data/raw/                                read-only symlink → shared corpus (nested Committee/[WP/]Session/)
data/protocols/pdf/      <stem>_Protokoll.pdf   A0 → symlinks to raw protocol PDFs (input)
data/transcripts/audio/  <stem>_Transkript.mp3  A0 → symlinks to raw session MP3s (input)
data/transcripts/manifest.txt            A0 → staged audio paths of trainable sessions
data/ingest_report.tsv                   A0 → per-session stem/flags audit
data/protocols/md/                       A → markdown protocols
data/protocols/md_clean/   (+ cover/)    A → cleaned bodies + separated cover pages
data/transcripts/md/                     A → diarised transcripts (<SD-SPK>SPEAKER_NN, timestamps)
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


## Running on SLURM (cluster specifics)

The GPU sbatch scripts (`transcribe*.sbatch`, `train_lora.sbatch`) already set
`--account=aisc --partition=aisc-batch`. You may use a different configuration on your cluster.

CPU-only work (e.g. the optional `pdf_to_markdown.sbatch`) runs under the **`default`
account + `normal` QOS** on `cpu-batch` (the `aisc` account/QOS is GPU-partitions
only); those scripts set that themselves. `HF_HOME` defaults to shared project
storage so large model downloads stay off the home quota. `uv sync` once to install.


## A0. Ingest the raw corpus (stage the nested delivery → flat layout)

The delivered corpus is nested `Committee/[Wahlperiode/]Session/` folders, each
with a protocol **PDF** + a session **MP3** and inconsistent file names. Point
`data/raw` at it once:

```bash
ln -s /sc/projects/sci-aisc/pilotproject-automatic-protocols/data2 data/raw
```

`ingest_corpus.py` walks `data/raw`, derives a canonical session stem
`<ABBR>[_<WP>]_<NN>` (committee abbreviation from the top folder's parens, optional
Wahlperiode, sitting number) and creates **relative symlinks** into the flat
layout — `data/protocols/pdf/<stem>_Protokoll.pdf` and
`data/transcripts/audio/<stem>_Transkript.mp3` — plus `data/transcripts/manifest.txt`
(audio of the trainable sessions, i.e. those with both a PDF and audio) and
`data/ingest_report.tsv` (per-session audit with flags `MISSING_PDF` /
`MISSING_AUDIO` / `MULTIPART(n)` / `COLLISION_WITH:…`).

```bash
uv run python scripts/ingest_corpus.py            # --dry-run to preview, --overwrite to re-link
```

Inspect `data/ingest_report.tsv` before converting — flagged rows are skipped from
the manifest. Multi-part audio (e.g. `SLausitz_3`) is staged as
`<stem>_Transkript.pt01.mp3`, `.pt02.mp3`, …; after transcription, merge the parts
back into one transcript:

```bash
uv run python scripts/ingest_corpus.py --merge-parts --transcript-dir data/transcripts/md
```


## A. Data conversion & cleaning

All converters take `--input` (a file **or** a directory) and `--out-dir`, skip existing outputs unless `--overwrite`, and use exit codes 0/1/2.

```bash
# 1) protocol PDF → markdown (Docling, layout + table aware; OCR off, born-digital)
uv run python scripts/pdf_to_markdown.py --input data/protocols/pdf --out-dir data/protocols/md
#    Docling is memory-hungry; a full corpus OOM-kills on the login node. For many/large
#    PDFs run it on a compute node instead (idempotent — skips already-converted files):
#    INPUT=data/protocols/pdf OUT_DIR=data/protocols/md sbatch scripts/pdf_to_markdown.sbatch

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

> **TOP boundaries (B1):** `tag_transcript_tops.py` also handles two tricky cases. Jointly-handled items ("TOP 4 gemeinsam mit TOP 5") may share an anchor instead of being mis-placed; and when one turn both *closes* the previous TOP (vote / "ich schließe den Tagesordnungspunkt") and *opens* the next, a follow-up LLM call splits the turn at the right segment line so the closing/vote stays with the previous TOP (both halves keep the speaker). Disable with `--no-boundary-split`.

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
| `OUT_DIR`     | auto `results/YYYYMMDD-HHMMSS` (4-bit smoke → `results/smoke_lora`) |
| `BASE_MODEL`  | `google/gemma-4-E2B-it`              |
| `MAX_SEQ_LEN` | auto = model context window (empty)  |
| `BITS`        | `16` (bf16 LoRA; `4` = QLoRA smoke)  |
| `EPOCHS`      | `3`                                  |
| `MAX_STEPS`   | `-1` (use epochs; set small to smoke-test) |
| `USE_CCE`     | `0` (set `1` for long-context — see below) |
| `LR`          | (empty = `2e-4`)                     |

The sbatch also sets `#SBATCH --qos=aisc` (faster AISC queueing), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (allocator fragmentation), and `PYTHONUNBUFFERED=1` (stream the per-step loss to the `.out` log). Each run writes a `train_log.md` **and** a `README.md` of its settings into the timestamped output dir.

Scale up to the 31B with `BASE_MODEL=google/gemma-4-31B-it`. `HF_HOME` defaults to shared project storage to keep downloads off the home quota.

Key `train_lora.py` flags: `--bits {4,16}`, `--cce`, `--lora-r/--lora-alpha/--lora-dropout`, `--target-modules`/`--exclude-modules`, `--epochs`, `--max-steps`, `--lr`, `--batch-size`/`--grad-accum`, `--max-seq-len`, `--packing`, `--attn {sdpa,flash_attention_2,eager}`.

> **Watch the sequence length:** per-TOP records with timestamps can exceed `MAX_SEQ_LEN=4096` real tokens and get truncated (losing part of the target). Raise `--max-seq-len` (e.g. 8192) for long sessions.
>
> **Long sequences on the 31B — use `USE_CCE=1`:** gemma-4-31B-it has a ~262k-token vocabulary, so the *stock* loss step materialises a `seq_len × vocab` logits tensor (plus an fp32 cross-entropy copy) on one device — this OOMs an 80 GB H100 well before the context window (historically ~32k borderline, `>40k` OOM). `--cce` (env `USE_CCE=1`) fixes it: it computes the loss with **cut-cross-entropy**, which never materialises the full logits tensor, so long and even **uncapped** training works. Numerically stable at long seq and in bf16, and it honours Gemma's logit softcap (verified at bf16@4096 and 4-bit@32768). The remaining memory cost at very long lengths is attention+activations, not logits — `--attn flash_attention_2` and/or more GPUs (`device_map=auto` shards across them) bound that. (An earlier liger fused-CE attempt diverged with NaN gradients at long seq/bf16 and was removed.)


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
