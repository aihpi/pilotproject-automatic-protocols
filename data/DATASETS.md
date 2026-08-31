# Dataset ledger

Tracks the training-dataset variants built by `scripts/build_dataset.py` (stage C of the
prep pipeline) and the variation between builds. `data/` is gitignored, so this file is
the local record of how each `data/train_*` folder was produced.

Pipeline: `raw → ingest → md(+pdf) → md_top (B1: tag_transcript_tops) →
md_prepared (B2: match_speakers) → train_* (C: build_dataset)`. Only **stage C** is
re-run for a prompt change; B1/B2 (LLM-driven TOP tagging + speaker resolution) already
incorporate the joint-TOP fix in the current `data/transcripts/md_prepared`.

## Full-corpus enlarged build (2026-07-01) — CURRENT

All committees ingested for WP7 (alongside existing WP8): the WP7 superset plus the later
Nextcloud drops — AIK, ARD (RA), ABJS, then AIL, ALEUV, ASGIV — for **325 paired sessions**.
The ~207 new sessions were transcribed + diarised (Whisper large-v3 + pyannote), their protocol
PDFs converted (`pdf_to_markdown`), TOP-tagged (B1) and speaker-resolved corpus-wide
(B2, `--transcript-llm`). Run through the dependency-chained sbatch scripts
(`tag_transcript_tops.sbatch` → `match_speakers.sbatch` → `build_dataset.sbatch`). Pinned
per-session hash split; `test/` eval sessions held out (verified 0 leakage in train/val).

**`data/train/cap65k/` is THE dataset** (per-top, with-document-fallback, transcript-llm).

| dataset | path | cap | train | val | sessions | notes |
|---|---|---|---|---|---|---|
| **cap65k** | `data/train/cap65k/` | 65536 | 1122 | 106 | 325 (31 val) | per-top + 74 untagged→whole-doc; transcript-llm speaker recovery |

Build (env-parameterised sbatch; secrets sourced from the home `.env` at submit — see script headers):

```
sbatch scripts/tag_transcript_tops.sbatch                                     # B1 (incremental; skips already-tagged)
OVERWRITE=1 TRANSCRIPT_LLM=1 sbatch scripts/match_speakers.sbatch             # B2 corpus-wide -> md_prepared_tllm + data/exclusions/exclusions_tllm.json
OVERWRITE=1 EXCLUSIONS=data/exclusions/exclusions_tllm.json \
  sbatch scripts/build_dataset.sbatch                                          # C -> data/train/cap65k
```

Build drops: 487 speaker-excluded TOPs, 10 target-too-short, 7 length-excluded (>65536 tok);
29 val records over the 8192-token val cap routed to train. Exclusions now live in `data/exclusions/`.

**Known speaker-ID imperfections (inspected + accepted 2026-07-01, recall over precision):** B2
leaves ~894 partial/non-schema names — all from the transcript-llm tier, guest surnames such as
"Herr Westphal" / "Frau Buttke" — and ~228 labels sit on cue-conflicted (under-segmented pyannote)
clusters, so a minority of turns carry a wrong or partial speaker. Kept as-is. Gold protocol targets
retain the real "Abgeordnete(r) Lastname (Fraktion)" wording; ASR-misheard surnames in the transcript
prose are left uncorrected.

Adapters: `results/YYYYMMDD-HHMMSS/`, mapping in `results/OVERVIEW_tllm.tsv`. The prior 2026-06-26
build (583/74, WP7 superset only) is **superseded** by this enlarged build — but note the 31B adapter
`results/20260626-115353` (job 2293906) was trained on that earlier 583/74 set, and the 12B run
(job 2293907) failed on the transformers / `gemma4_unified` issue (GitHub issue #10). Retraining on
this 1122/106 set is the next step once approved.

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
