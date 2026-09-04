<div style="background-color: #ffffff; color: #000000; padding: 10px;">
<img src="00_aisc/img/logo_aisc_bmftr.jpg">
<h1> Automatic Protocols: committee transcripts to protocols
</div>

This repository trains and evaluates a LoRA adapter for `google/gemma-4-31B-it` that turns the diarised transcript of a Landtag committee sitting into the corresponding section of a German committee protocol, one agenda item (Tagesordnungspunkt, TOP) at a time. It holds the complete data pipeline (ingest, transcription, agenda tagging, speaker resolution, dataset build), the training and evaluation scripts, and the documentation of the model that is in production in the sister app [pilotproject-protokollierungsassistenz](https://github.com/aihpi/pilotproject-protokollierungsassistenz) as the "Landtagstil" configuration.

## What this repository contains, and what it does not

The repository ships code and documentation only. The training corpus (audio recordings and protocol PDFs of the Landtag Brandenburg committees), every derived dataset, the held-out evaluation transcripts and the trained adapter weights are not part of it: the corpus is not redistributable, and the derived files contain it verbatim. `data/`, `results/` and `test/*/` are gitignored; only their ledgers (`data/DATASETS.md`, `results/README.md`, `test/manifest.tsv`) are tracked. To reproduce the work you need access to the corpus and an HPI cluster account, or your own SLURM cluster and the adjustments described below.

## The model

| | |
|---|---|
| Run | `results/20260902-31b_cap48k` (SLURM job 2502986) |
| Base model | `google/gemma-4-31B-it`, loaded as `unsloth/gemma-4-31b-it-unsloth-bnb-4bit` (4-bit QLoRA) |
| Adapter | LoRA r = 8, alpha = 8, dropout 0, all linear projections |
| Sequence cap | 49,152 tokens (records above the cap are excluded at build time) |
| Data | `data/train/cap48k`: 1,115 training and 106 validation records from 320 sittings |
| Training | 3 epochs, lr 2e-4, batch 1 x 4 accumulation, early stopping on validation loss, one H100 80 GB, about 19.5 h |
| Result | validation loss 0.6927 (best checkpoint) |

The cap was chosen by a sweep over otherwise identical runs:

| dataset | cap (tokens) | train / val | validation loss |
|---|---|---|---|
| cap32k | 32,768 | 1,100 / 106 | 0.6936 |
| cap40k | 40,960 | 1,110 / 106 | 0.6942 |
| **cap48k** | **49,152** | **1,115 / 106** | **0.6927** |
| cap64k | 65,536 | 1,122 / 106 | CUDA out of memory at step 96 |

The 65k run fails because the loss materialises a `sequence x 262,144` vocabulary logits tensor; on one 80 GB H100 the fused Unsloth cross-entropy carries records up to about 50k tokens. Details of the runs are in `results/README.md`, of the datasets in `data/DATASETS.md`. A report written for readers outside the project, covering the corpus, the preparation pipeline, the quality filters and the training configuration, is in [`docs/training-report.md`](docs/training-report.md).

## Setup

```
uv sync
cp .env_example .env
```

Fill in `.env`: `HF_TOKEN` (gated gemma tokenizer and the pyannote diariser) and `OPENAI_API_KEY` plus `OPENAI_API_BASE` (the gpt-oss-120b endpoint used by the LLM stages B1 and B2). Every sbatch script sources `.env`. Do not commit it.

The Unsloth trainer and evaluator run from a standalone venv, because Unsloth's pins conflict with the base environment. Build it once on a compute node:

```
uv venv .venv-unsloth --python 3.12
uv pip install --python .venv-unsloth unsloth trl peft datasets bitsandbytes tensorboard
```

The `.sbatch` scripts under `scripts/` are written for the HPI sc cluster and require SLURM. Every line that a different cluster must change (account, QOS, partition, GPU type, node constraints, walltime) is marked `# ADJUST:`. `HF_HOME` defaults to `<repo>/hf-cache`; set it in `.env` to point at shared storage. On the HPI cluster the corpus lives on shared project storage and `data/raw` is a symlink to it.

## Workflow

Each stage writes a new sub-folder under `data/`, so intermediates can be inspected. Stages A and B only run when new recordings arrive; a prompt or cap change needs stage C onwards.

### A. Ingest and convert

New deliveries arrive as a flat folder of `<sitting>. <ABBR> vom <DD.MM.YYYY>.{mp3,pdf}` files. `import_dropzip.py` reshapes such a drop into the nested corpus layout, `ingest_corpus.py` stages the corpus into flat `data/protocols/pdf/` and `data/transcripts/audio/` symlinks, keeps a ledger of what it has seen and writes `data/transcripts/manifest_new.txt` with the sessions that are new or changed since the last run.

```
uv run python scripts/import_dropzip.py import --drop-dir /path/to/Daten --dry-run
uv run python scripts/import_dropzip.py import --drop-dir /path/to/Daten
uv run python scripts/ingest_corpus.py
INPUT=data/protocols/pdf OUT_DIR=data/protocols/md sbatch scripts/pdf_to_markdown.sbatch
INPUT_LIST=data/transcripts/manifest_new.txt OUT_DIR=data/transcripts/md DIARIZE=1 sbatch scripts/transcribe.sbatch
```

Protocols become markdown through Docling; audio becomes a diarised transcript (WhisperX large-v3 plus pyannote) with `<SD-SPK>SPEAKER_NN</SD>` tags and timestamps. For many files use `transcribe_array.sbatch` (one GPU per file, launch instructions in its header). Multi-part recordings are merged with `ingest_corpus.py --merge-parts`.

### B. Prepare: agenda tags and speaker names

Two LLM stages turn a raw transcript into training input. B1 reads the agenda from the protocol cover and inserts `<SD-TOP>TOP N</SD>` markers where the chair takes up each item. B2 maps every `SPEAKER_NN` to a name and role, using the attendance list, the protocol prose and, in the transcript-LLM tier, the transcript itself; items that still contain an unidentified speaker are written to an exclusions file.

```
sbatch scripts/tag_transcript_tops.sbatch
OVERWRITE=1 TRANSCRIPT_LLM=1 sbatch scripts/match_speakers.sbatch
```

Output: `data/transcripts/md_prepared_tllm/` (final transcripts), `data/speaker_maps_tllm/` (per-session reports), `data/exclusions/exclusions_tllm.json`. Both scripts also run directly with `uv run python scripts/<name>.py --help`; pass `--no-llm` for the regex fallback.

### C. Build the dataset

`build_dataset.py` pairs prepared transcripts with protocols per TOP, cleans the protocol (`preprocess_protocol.clean_protocol`), drops excluded items, keeps the sessions in `test/manifest.tsv` out of both splits, splits the rest by session and writes chat-format JSONL. It writes length exclusions back into the exclusions file it is given, so build from a copy:

```
set -a; . .env; set +a
cp data/exclusions/exclusions_tllm.json data/exclusions/exclusions_<name>.json
OUT_DIR=data/train/<name> MAX_SEQ_LEN=49152 EXCLUSIONS=data/exclusions/exclusions_<name>.json sbatch --export=ALL scripts/build_dataset.sbatch
```

Each record is `{"messages": [system, user, assistant], "meta": {...}}`. The user turn is built by `scripts/utils/prompt_io.py`, the single source of truth shared with the app: tags and timestamps are stripped to `Name: utterance` lines and wrapped in the `TOP: ... / Transkript: ... / Zusammenfassung:` frame; the system prompt is `scripts/utils/prompt_summarize.txt`. `meta.seq_tokens` is the real token count; `src_tokens` and `tgt_tokens` are word counts.

### D. Train

```
TRAIN_JSONL=data/train/cap48k/train.jsonl VAL_JSONL=data/train/cap48k/val.jsonl MAX_SEQ_LEN=49152 OUT_DIR=results/$(date +%Y%m%d)-31b_cap48k sbatch scripts/train_lora_unsloth.sbatch
```

The run folder holds the adapter, tokenizer, `train_log.md` and TensorBoard logs. Hyperparameters are overridable through the environment (`LORA_R`, `LORA_ALPHA`, `LR`, `EPOCHS`, `ES_PATIENCE`, see the script).

### E. Evaluate

`eval_lora.py` summarises every example in a directory with one adapter and two decode presets (plain sampling and a mild repetition penalty). `test/` is the held-out set; `eval_report.py` compares the outputs with the gold protocols and writes `COMPARISON.md`.

```
FRAMEWORK=unsloth ADAPTER=results/20260902-31b_cap48k ADAPTER_ID=cap48k MAX_SEQ_LEN=49152 RUN_DIR=data/test/$(date +%Y%m%d-%H%M%S) sbatch scripts/eval_lora.sbatch
uv run python scripts/eval_report.py data/test/<timestamp>
```

The held-out transcripts are regenerated with `uv run python scripts/build_test_set.py --source-dir <dir>` (see `test/README.md`).

### Generating protocols for new transcripts

Any directory of example folders works as `EXAMPLES_DIR`, each folder holding one `<stem>_Transkript.md` with `<SD-TOP>` markers; a gold protocol is optional. Run stage E with `EXAMPLES_DIR=<dir>`; the output is one markdown file per adapter and decode preset with a `## Zu TOP n:` section per item. In production the app performs the same per-TOP call against the adapter served on the LiteLLM hub.

## Data layout

```
data/raw/                              symlink to the shared corpus (Committee/[WP/]Session/)
data/protocols/pdf/, data/protocols/md/     staged protocol PDFs and their markdown
data/transcripts/audio/, data/transcripts/md/   staged audio and diarised transcripts
data/transcripts/md_top/               B1: transcripts with <SD-TOP> tags (+ top_reports/)
data/transcripts/md_prepared_tllm/     B2: speaker names resolved (dataset input)
data/speaker_maps_tllm/, data/exclusions/    B2 reports and exclusion files
data/train/<name>/{train,val}.jsonl    C: one folder per dataset build
data/test/<timestamp>/                 E: evaluation outputs
results/<run>/                         D: adapters (results/README.md lists them)
test/<example>/                        held-out transcripts and gold (regenerated, untracked)
```

## Known data limitation

The protocol targets of `cap48k` carry whitespace artefacts from PDF extraction: runs of several spaces inside justified lines (about 7 % of word gaps, present in 92 % of records) and a few hundred words split at former line breaks (`inner- halb`). The shipped adapter reproduces the multi-space runs. `preprocess_protocol.py` now removes both, and `data/train/cap48k_clean` is the rebuilt set; the adapter has not yet been retrained on it. Genuine typos in the gold protocols are rare (about one per 65,000 words) and are left as they are.

## Limitations

The pipeline is built for German committee protocols of the Landtag Brandenburg; the agenda tagging and the speaker resolution assume that structure. Party affiliations and Drucksache numbers in the output must be checked against the source: the diariser leaves most speakers unnamed and the model has been observed to add both. Very long items (above roughly 20,000 words of transcript) can end in repetitions or an unfinished sentence.

## Contact

Questions and installation support: [kisz@hpi.de](mailto:kisz@hpi.de).

## Authors

- [Hanno Müller](https://github.com/hanno-mueller-hpi)

## Licence

This project is licensed under the MIT licence, see [LICENSE](LICENSE).

---

## Acknowledgements
<img src="00_aisc/img/logo_bmftr_de.png" alt="drawing" style="width:170px;"/>

The [AI Service Centre Berlin Brandenburg](http://hpi.de/kisz) is funded by the [Federal Ministry of Research, Technology and Space](https://www.bmbf.de/) under the funding code 16IS22092.
