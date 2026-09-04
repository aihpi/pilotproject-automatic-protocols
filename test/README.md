# `test/`: held-out evaluation set

Six committee sittings of the Landtag Brandenburg that `build_dataset.py --holdout-manifest test/manifest.tsv` keeps out of every training and validation split. Scores on them measure generalisation. Only `manifest.tsv` and this file are tracked; the transcripts and gold protocols contain the corpus verbatim and are regenerated locally:

```
uv run python scripts/build_test_set.py --source-dir <dir>
```

`<dir>` holds `transcripts/md_prepared/<stem>_Transkript.md` and `protocols/md/<stem>_Protokoll.md` for the six stems (on the HPI cluster: `archive/data_example02` in the projects clone). The gold protocols are run through the same `clean_protocol` as the training targets, so the manifest's `gold_words` column changes when the cleaner changes.

## Layout

```
test/<example>/<stem>_Transkript.md   input (unresolved SPEAKER_NN rewritten to placeholders)
test/<example>/<stem>_Protokoll.md    gold (cover and footers stripped)
test/<example>/speaker_map.tsv        audit: SPEAKER_NN to placeholder
test/manifest.tsv                     per-example metadata (tracked)
```

Evaluation outputs go to the gitignored `data/test/<timestamp>/` (`RUN_DIR` of `eval_lora.sbatch`); `eval_lora.sbatch` reads `EXAMPLES_DIR=test` by default.

## Sessions

| example | stem | TOPs | speakers | invented | transcript words |
|---|---|---|---|---|---|
| short_ARD_1 | ARD_1 | 6 | 5 | 0 | 4.1k |
| mid_HA_1 | HA_1 | 11 | 11 | 2 | 11.1k |
| mid_HA_3 | HA_3 | 1 | 7 | 5 | 8.7k |
| mid_ABJS_5 | ABJS_5 | 1 | 11 | 5 | 17.4k |
| long_SBue_2 | SBue_2 | 3 | 18 | 9 | 36.8k |
| long_SBue_3 | SBue_3 | 2 | 21 | 17 | 32.8k |

The plenary sitting `Plenum_8-31` (structurally different, about 65k words) is deliberately excluded.

## Invented speaker names

Unresolved diarisation tags `<SD-SPK>SPEAKER_NN</SD>` are rewritten to obviously synthetic placeholders such as `Max Mustermann (ABC)` (consistent within a transcript, assigned by first appearance). Resolved real names are kept. This makes the input self-consistent and tests fabrication resistance: a faithful model echoes the placeholder rather than recalling a real name. For rewritten turns the transcript names therefore do not match the gold protocol, so judge those turns by whether the output uses the placeholder, not by gold overlap.
