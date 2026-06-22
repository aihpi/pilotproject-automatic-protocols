# `test/` — clean held-out evaluation set

Tracked, uncontaminated evaluation examples for the transcript→protocol LoRA
adapters. Regenerate with `tmp/build_test_set.py` (run from repo root).

Every session here is **absent from all training splits** (`no_docs` + `with_docs`,
`train` + `val` of the 65k builds), so scores here measure generalisation. This
replaces the old `data/test/examples/` set, where `ARD_7_47` / `AIL_12` had leaked
into training (see `data/test/20260619-093916/COMPARISON.md`).

## Layout

```
test/<example>/<stem>_Transkript.md   # input  (SPEAKER_NN → invented names)
test/<example>/<stem>_Protokoll.md    # gold   (cover + page-footers stripped)
test/<example>/speaker_map.tsv        # audit: SPEAKER_NN → placeholder
test/manifest.tsv                     # per-example metadata
```

Eval run outputs still go to the gitignored `data/test/<YYYYMMDD-HHMMSS>/`; only
these inputs are tracked. The harness reads `test/` by default.

## Sessions (all committee, Brandenburg Landtag)

| example | stem | TOPs | speakers | invented | ~words |
|---|---|---|---|---|---|
| short_ARD_1 | ARD_1 | 6 | 5 | 0 | 4.1k |
| mid_HA_1 | HA_1 | 11 | 11 | 2 | 11.1k |
| mid_HA_3 | HA_3 | 1 | 7 | 5 | 8.7k |
| mid_ABJS_5 | ABJS_5 | 1 | 11 | 5 | 17.4k |
| long_SBue_2 | SBue_2 | 3 | 18 | 9 | 36.8k |
| long_SBue_3 | SBue_3 | 2 | 21 | 17 | 32.8k |

The plenary `Plenum_8-31` (structurally different, ~65k words near the cap) and the
contaminated `SLausitz_1/2` (in training) are deliberately excluded.

## Invented speaker names — what they test

Unresolved diarisation tags `<SD-SPK>SPEAKER_NN</SD>` are rewritten to obviously-
synthetic placeholders, e.g. `Max Mustermann (ABC)`, `Erika Musterfrau (DEF)`
(consistent within a transcript, assigned by first appearance). Resolved real names
(`Kristy Augustin (CDU)`) are kept.

This makes the input self-consistent (no raw labels) and tests **fabrication-
resistance**: a faithful model should echo the placeholder, not invent or recall a
real name. Consequence: for rewritten turns the transcript names do **not** match the
gold protocol's real names, so gold name-overlap is N/A there — judge those turns by
whether the output uses the placeholder, not by gold match.
