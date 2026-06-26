# Dataset ledger

Tracks the training-dataset variants built by `scripts/build_dataset.py` (stage C of the
prep pipeline) and the variation between builds. `data/` is gitignored, so this file is
the local record of how each `data/train_*` folder was produced.

Pipeline: `raw → ingest → md(+pdf) → md_top (B1: tag_transcript_tops) →
md_prepared (B2: match_speakers) → train_* (C: build_dataset)`. Only **stage C** is
re-run for a prompt change; B1/B2 (LLM-driven TOP tagging + speaker resolution) already
incorporate the joint-TOP fix in the current `data/transcripts/md_prepared`.

## Superset + transcript-LLM build (2026-06-26) — CURRENT

Full WP7 superset ingested (`Daten_full.zip`: 144 sessions across 6 committees AHF/HA/AWFK/AEE/
SLausitz/AHK; the earlier 32-session `Daten.zip` is a subset). New WP7 sessions transcribed +
diarised, TOP-tagged (B1, with the bare-number/wrapped-agenda parser fix) and speaker-resolved
(B2). B2 was then re-run corpus-wide with the new transcript-driven tier
(`match_speakers.py --transcript-llm`): corpus speaker resolution rose **51% → 74%** (+633 labels,
606 via transcript-llm), recovering guests/experts absent from the protocol. Pinned per-session
split (`build_dataset.py` hash split) and the `test/` eval sessions held out.

**`data/train_cap65k/` is THE dataset** (the per-top, with-document-fallback, transcript-llm
build). Every other `data/train_*` folder was experimental and has been moved to
`data/archive/` (16 dirs: caps 4k/16k/32k/uncapped, no-docs variants, footer-clean, wp7
pre-transcript-llm, etc.).

| dataset | path | cap | train | val | notes |
|---|---|---|---|---|---|
| **train_cap65k** | `data/train_cap65k/` | 65536 | 583 | 74 | per-top + 15 untagged-as-whole-document; transcript-llm speaker recovery |

Build commands (the `with_docs` recipe; the folder was renamed `…_with_docs_cap65k_tllm` → `train_cap65k`):

```
uv run python scripts/match_speakers.py --transcript-dir data/transcripts/md_top \
  --protocol-dir data/protocols/md --out-transcript-dir data/transcripts/md_prepared_tllm \
  --report-dir data/speaker_maps_tllm --exclusions-out data/exclusions_tllm.json \
  --transcript-llm --overwrite --concurrency 2     # low concurrency: gpt-oss throttles parallel calls
cp data/exclusions_tllm.json data/exclusions_tllm_build.json
uv run python scripts/build_dataset.py --transcript-dir data/transcripts/md_prepared_tllm \
  --protocol-dir data/protocols/md --exclusions data/exclusions_tllm_build.json \
  --granularity per-top --max-seq-len 65536 --include-untagged-as-document \
  --holdout-manifest test/manifest.tsv --out-dir data/train_cap65k --overwrite
```

Adapters: `results/YYYYMMDD-HHMMSS/`, mapping in `results/OVERVIEW_tllm.tsv`
(gemma-4-31B-it + gemma-4-12B-it, both on `train_cap65k`).

## Footer-cleaned 65k variants (built 2026-06-22) — superseded

Rebuilt after fixing `preprocess_protocol.clean_protocol` to strip the leaks found in the
2026-06-18 targets: plain-text/heading page footers (`HA` / `## N. (öffentliche) Sitzung` /
date) and the post-ratification `(Dieses Protokoll wurde … § 83 … bestätigt.)` line (84/170
and 22/170 of the old no_docs targets were affected). Same prompt `380b3896`, same inputs.
**Old `*_cap65k` folders kept for comparison** (they still contain the footer leaks).

| variant | path | cap | prompt | train | val | footer leaks | notes |
|---|---|---|---|---|---|---|---|
| no_docs cap65k clean  | `data/train_no_docs_cap65k_clean/`   | 65536 | `380b3896` | 167 | 24 | 0 | per-top only |
| with_docs cap65k clean| `data/train_with_docs_cap65k_clean/` | 65536 | `380b3896` | 179 | 14 | 0 | per-top + whole-doc fallback (6 untagged); 3 length-excl >65k |

Build commands (note `--transcript-dir data/transcripts/md_prepared` is required — the
default `data/transcripts/md` is untagged → "no aligned TOPs"; `--overwrite` to rewrite):

```
UV_NO_SYNC=1 .venv/bin/python scripts/build_dataset.py \
  --transcript-dir data/transcripts/md_prepared --protocol-dir data/protocols/md \
  --exclusions data/exclusions_build.json --granularity per-top --max-seq-len 65536 \
  --out-dir data/train_no_docs_cap65k_clean --overwrite
# with_docs: add --include-untagged-as-document, --out-dir data/train_with_docs_cap65k_clean
```

Record-count drop vs the 2026-06-18 build (170→167 no_docs) is from a few targets falling
below `--min-tgt-tokens` once their footer padding was removed.

## Previous 65k variants (rebuilt 2026-06-18, commit `7acd626`) — superseded, kept for A/B

Rebuilt to bake in the **finalized hardened system prompt** (sha `380b3896`, replaces the
earlier `268ac54f`). Inputs: `--transcript-dir data/transcripts/md_prepared`
`--protocol-dir data/protocols/md` `--exclusions data/exclusions_build.json`
`--granularity per-top`. Tokeniser: `google/gemma-4-31B-it`.

| variant | path | cap | prompt sha | train | val | length-excl (>cap) | notes |
|---|---|---|---|---|---|---|---|
| no_docs cap65k  | `data/train_no_docs_cap65k/`   | 65536 | `380b3896` | 170 | 24 | 0 | per-top only; max real seq 29 634 tok (all fit) |
| with_docs cap65k| `data/train_with_docs_cap65k/` | 65536 | `380b3896` | 182 | 14 | 3 | per-top + whole-doc fallback for 6 untagged sessions (3 dropped >65k) |

Build commands:

```
uv run python scripts/build_dataset.py --transcript-dir data/transcripts/md_prepared \
  --protocol-dir data/protocols/md --exclusions data/exclusions_build.json \
  --granularity per-top --max-seq-len 65536 --out-dir data/train_no_docs_cap65k --overwrite

uv run python scripts/build_dataset.py --transcript-dir data/transcripts/md_prepared \
  --protocol-dir data/protocols/md --exclusions data/exclusions_build.json \
  --granularity per-top --max-seq-len 65536 --include-untagged-as-document \
  --out-dir data/train_with_docs_cap65k --overwrite
```

(`build_dataset.py` merges any new length-exclusions into `data/exclusions_build.json`;
the with_docs build added 3.)

## Variation vs the previous (v1) build of these folders

| variant | prev (2026-06-16) | now (2026-06-18) | what changed |
|---|---|---|---|
| no_docs cap65k  | 175 train / 24 val, prompt `268ac54f` | 170 / 24, prompt `380b3896` | hardened prompt; ~5 fewer records (current `md_prepared`/exclusions state — more speaker-excluded TOPs) |
| with_docs cap65k| 187 train / 14 val, prompt `268ac54f` | 182 / 14, prompt `380b3896` | same |

The drop is **not** from length (no_docs: 0 length-excluded; max seq 29 634 < 65 536) — it
reflects the current speaker-exclusion set (`250` TOPs excluded). These v2 datasets are
what the 65k retrain (results/ timestamped runs) consumes; each run's `train_log.md`
records the dataset path + prompt sha, so adapter↔dataset provenance is preserved.

## Other dataset folders (not rebuilt; kept as-is)

`data/train/`, `data/train_no_docs[_16k]/`, `data/train_with_docs[_16k]/`,
`data/train_*_uncapped/` — earlier variants (caps 4k/16k/32k/uncapped, prompt `268ac54f`).
Left untouched; superseded by the 65k v2 builds for the current retrain.
