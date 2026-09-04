# Dataset ledger

`data/` is gitignored, so this file is the record of how each `data/train/<name>/` folder was produced by `scripts/build_dataset.py` (stage C). Pipeline: raw corpus, ingest, markdown (A), `md_top` (B1 `tag_transcript_tops.py`), `md_prepared_tllm` (B2 `match_speakers.py --transcript-llm`), `train/<name>` (C). A cap or prompt change only re-runs stage C.

## cap48k (built 2026-09-02): the production dataset

The adapter in production (`results/20260902-31b_cap48k`) was trained on this set.

| | |
|---|---|
| Inputs | `data/transcripts/md_prepared_tllm/` + `data/protocols/md/` (cleaned at build) |
| Exclusions | copy of `data/exclusions/exclusions_tllm.json` (B2 speaker exclusions; length exclusions appended by the build) |
| Hold-out | `--holdout-manifest test/manifest.tsv` (six sittings kept out of both splits) |
| Granularity | per TOP, plus one whole-document record for sittings without aligned TOP tags (`--include-untagged-as-document`) |
| Cap | 49,152 tokens per record (gemma tokenizer); validation records above 8,192 tokens go to train |
| Split | by session, seed 42: 320 sittings, 31 of them validation |
| Records | 1,115 train / 106 val |
| Dropped | 10 targets too short, 488 TOPs with an unresolved speaker, 13 over the cap, 74 sittings without aligned TOPs (their document record was used instead), 29 val records moved to train for length |

```
set -a; . .env; set +a
cp data/exclusions/exclusions_tllm.json data/exclusions/exclusions_cap48k.json
OUT_DIR=data/train/cap48k MAX_SEQ_LEN=49152 EXCLUSIONS=data/exclusions/exclusions_cap48k.json sbatch --export=ALL scripts/build_dataset.sbatch
```

Sweep siblings built the same way with a different `MAX_SEQ_LEN`: `cap32k` (1,100 / 106), `cap40k` (1,110 / 106), `cap65k` (1,122 / 106, the 65k run OOMs). Their adapters are listed in `results/README.md`.

Speaker resolution (B2) favours recall: several hundred labels carry a partial name such as "Herr Westphal", and some under-segmented diarisation clusters put a wrong speaker on a minority of turns. Gold targets keep the protocol's own wording ("Abgeordnete(r) Lastname (Fraktion)").

## cap48k_clean (built 2026-09-04): rebuilt with the whitespace fix, not yet trained on

`preprocess_protocol.clean_protocol` now collapses the multi-space runs that Docling leaves in justified text, rejoins words split at former line breaks (`inner- halb`) and drops bare `Landtag Brandenburg` / `P-XX n/m` footer lines. `cap48k_clean` is the cap48k recipe run through the fixed cleaner (exclusions copy `exclusions_cap48k_clean.json`, SLURM job 2506171): the same 320 sittings and the same 1,115 / 106 split; 9 records over the cap instead of 13 and 492 speaker-excluded instead of 488, because shorter targets moved four records between the two exclusion classes. In the training targets the artefacts fall from 1,028 records with multi-space runs, 127 split words, 10 `Landtag Brandenburg` lines and 3 protocol-number lines to zero of each. The production adapter predates this fix; retraining on `cap48k_clean` is the next training run.

## Superseded builds

Earlier datasets (`train_no_docs*`, `train_with_docs*`, 16k / 32k / 65k caps, the 2026-06 `cap65k` from 68 and later 229 sittings) live under `data/archive/` on the cluster and are documented in their folders' build logs. None of them is used any more.
