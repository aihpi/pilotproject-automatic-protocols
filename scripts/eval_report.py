#!/usr/bin/env python3
"""Build data/test/COMPARISON.md from the generated summaries.

Walks ``data/test/<example>/summaries/*.md``, parses each output's front matter,
computes degeneration metrics (eval_io.degeneration_metrics) and a length ratio
vs the gold protocol, and writes a Markdown report grouped by example plus a
baseline-vs-antirep delta per adapter. Pure-python; run any time after some
summaries exist:

    uv run python scripts/eval_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import eval_io

TEST_DIR = Path("data/test")               # run outputs: data/test/<YYYYMMDD-HHMMSS>/
EXAMPLES_DIR = Path("test")                # tracked clean held-out inputs + gold


def latest_run_dir() -> Path | None:
    """Most recent eval-run folder data/test/<YYYYMMDD-HHMMSS>/ (digit-led name)."""
    cands = sorted((p for p in TEST_DIR.iterdir()
                    if p.is_dir() and p.name[:1].isdigit()), reverse=True)
    return cands[0] if cands else None

# Static, human-written findings appended to the report (survives regeneration).
FINDINGS = """
## Findings & issues (for discussion)

> Partial: written while the 9 PEFT/FSDP contender jobs were still queued behind
> training. So far this covers the earlier hand-run eval (cce_withdocs `172746`,
> fsdp `001021`, unsloth `182728` on AIK_8_1 / AIL_6 / HA_8_4), the Keras
> gemma-3-1b demo, and one PEFT baseline on short_ARD_1. Numbers will firm up
> once every adapter has run on every example under identical baseline+antirep
> decoding.

### ⚠️ The metrics can be MISLEADING — eyeball the text
- **ts/tag/maxrep do NOT detect character-salad.** The `cce_nodocs_cap32k` *antirep*
  output scored ts=0/tag=0/maxrep=1 ("clean") but TOP 1 is literal gibberish
  (`Z U T O P N S J L … ouuuuuvvvwww… ==== ====`). `gib%` (share of salad tokens)
  was added to catch this, but it *averages*, so a single garbage TOP in a long
  output can still read low — **always read a sample**.
- **`ratio` is rough**: gold still contains cover/Anlagen, so ratios understate coverage.
- **`votes`/`ts` are heuristics** (`ts` counts `[HH:MM`/`(HH:MM`; `votes` needs spaced colons).

### ⚠️ antirep decoding — preset RETUNED (was too aggressive)
The original antirep preset (`repetition_penalty 1.3` + `no_repeat_ngram_size 3`) reduced
the metric-visible degeneration (timestamp leaks, loops) **but replaced them with
character-salad collapse**: once the model can't repeat legitimate German 3-grams
("Der Ausschuss", "Ja : Nein : …") it emits random letters/symbols (seen in
`cce_nodocs_cap32k` `short_ARD_1` antirep, which the ts/tag/maxrep metrics scored "clean").
→ The preset is now **gentle**: `repetition_penalty 1.15`, `no_repeat_ngram_size 0` (off).
Those v1 salad outputs are archived, not deleted. Re-judge by **reading**, not metrics.

### Confounds
- The AIK/AIL/HA outputs mix *different adapters and decode settings* — not yet
  apples-to-apples. The shortest-first full matrix (running) fixes this.
- Both baseline AND antirep degenerate in places: e.g. unsloth `long_singletop_AHF_7`
  *baseline* is also gibberish (gib≈27%). The 74k-token single TOP truncated to the cap
  is intrinsically hard regardless of decoding.

### Per-framework behaviour observed
- **FSDP (`001021`)** — cleanest by far (maxrep=1, few/zero timestamp leaks),
  best structure (`**Aus der Beratung**` / `**Beschluss**`, vote triples), despite
  the *worst* training loss (3.5). Strong evidence that **eval/train loss does not
  predict generation quality**.
- **cce_withdocs (`172746`)** — generally clean; one repetition wobble on AIK_8_1
  (maxrep=13). `with_docs` context seems to help entity fidelity.
- **Unsloth (`182728`)** — the degeneration case from the handoff: heavy timestamp
  echoing (ts=483 on AIL_6, 109 on AIK_8_1) and a 133-line repetition loop on
  HA_8_4; mangles place-names inconsistently (one place → 3 spellings). Its
  near-zero train loss (0.1152) signals **memorisation/overfit**, consistent with
  echoing the input.
- **Keras (gemma-3-1b)** — a *different, 1B* base: massive timestamp/tag echo
  (ts up to 1172). Exercises the Keras path but is **not a like-for-like contender**.

### Systemic issues (largely adapter-independent)
1. **Doubled `## Zu TOP n` heading** — the harness writes `## Zu TOP n` and the
   model also emits its own `## Zu TOP n: <title>`. Harness-side fix (drop one).
2. **Self-emitted zero/paren timestamps** `(00:00:00.000 --> ...)` and invented
   markers like `(Sitzungsbeginn …)` — not in the input; a learned habit. antirep
   decoding *does* cut them (they're repetitive n-grams) but risks salad (above); the
   durable fix is clean training targets + a post-processing strip.
3. **Whisper input loops** — "Vielen Dank." repeats up to **107×** (AHF_7), 61×
   (ARD_7_47), 21× (AIK_8_1). Clean adapters absorb them; Unsloth amplifies them
   into output loops. Fix: de-duplicate consecutive lines in the input.
4. **Vote-legend leak** — the placeholder `(Ja : Nein : Enthaltungen)` is sometimes
   emitted literally instead of counts (e.g. for unanimous votes).
5. **System prompt** says nothing about ignoring timestamps/tags/artifacts — a
   cheap lever to test, esp. for off-distribution inputs.

### Recommended fix order
1. **Retune the decode preset** — DONE: antirep is now `repetition_penalty 1.15`,
   `no_repeat_ngram_size 0` (the old `1.3`/`3` caused salad). Judge by reading.
2. Input cleaning: collapse consecutive duplicate (Whisper) lines before inference.
3. Output post-processing: strip timestamp lines, de-dupe `## Zu TOP` headings.
4. Training-side (now landed): **assistant-only loss** (validated) + the hardened
   prompt — both need a *retrain* (+ dataset rebuilt with the new prompt) to take effect.
5. System-prompt hardening A/B once retrained.
6. Only then: more/cleaner data, preferably `with_docs`.

_Net: decoding tweaks alone won't fix this cleanly — the strongest levers are the
training-side fixes (assistant-only masking, cleaner targets) plus light output
post-processing; treat metrics as a screen and confirm by reading the text._
"""


def parse_front_matter(text: str) -> tuple[dict, str]:
    meta: dict[str, str] = {}
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                if ": " in line and not line.startswith(" "):
                    k, v = line.split(": ", 1)
                    meta[k.strip()] = v.strip()
            return meta, text[end + 5:]
    return meta, text


def gold_words(example_name: str) -> int:
    g = sorted((EXAMPLES_DIR / example_name).glob("*_Protokoll.md"))   # symlinked gold
    if not g:                                                          # fallback: derive stem
        cand = Path("data/protocols/md") / f"{example_name}_Protokoll.md"
        if cand.exists():
            g = [cand]
    if not g:
        return 0
    _, body = parse_front_matter(g[0].read_text(encoding="utf-8"))
    return len(body.split())


def _framework_from_name(name: str) -> str:
    n = name.lower()
    if n.startswith("unsloth"):
        return "unsloth"
    if n.startswith("fsdp"):
        return "fsdp"
    if n.startswith("keras"):
        return "keras"
    return "peft"


def main() -> int:
    # Run folder: argv[1] if given, else the most recent data/test/<ts>/.
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run_dir()
    if not run_dir or not run_dir.is_dir():
        print("no eval-run folder under data/test/ (pass one as argv[1], "
              "e.g. data/test/20260619-101010)", file=sys.stderr)
        return 1
    rows = []  # (example, adapter, framework, decode, metrics, ratio)
    for ex in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        gw = gold_words(ex.name) or 1
        # flat scheme written by eval_lora: '<id>__<fw>__<gran>__<decode>.md'
        for f in sorted(ex.glob("*.md")):
            if f.name in ("README.md", "COMPARISON.md"):
                continue
            meta, body = parse_front_matter(f.read_text(encoding="utf-8"))
            m = eval_io.degeneration_metrics(body)
            adapter = meta.get("adapter_id", f.stem)
            rows.append({
                "example": ex.name,
                "adapter": adapter,
                "framework": meta.get("framework", _framework_from_name(f.stem)),
                "decode": meta.get("decode", "?"),
                "ratio": m["words"] / gw,
                **m,
            })

    if not rows:
        print(f"no summaries found under {run_dir}/<example>/*.md", file=sys.stderr)
        return 1

    out = ["# LoRA evaluation — output comparison",
           "",
           f"_Auto-generated by `scripts/eval_report.py` from {len(rows)} summaries._",
           "",
           "**Degeneration flags** (lower is better unless noted): `ts`=leaked "
           "`[HH:MM]` transcript timestamps, `tag`=leaked `<SD-…>` tags, "
           "`maxrep`=longest run of identical consecutive lines (≥3 ⇒ loop), "
           "`tops`=`## Zu TOP n` sections emitted, `votes`=vote-triples, "
           "`ratio`=output/gold word ratio.",
           ""]

    # group by example
    for ex in sorted({r["example"] for r in rows}):
        out += [f"## {ex}", "",
                "| adapter | fw | decode | words | ratio | ts | tag | maxrep | gib% | tops | votes |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted((r for r in rows if r["example"] == ex),
                        key=lambda r: (r["adapter"], r["decode"])):
            gib = r.get("gibberish_pct", 0.0)
            flag = " ⚠️" if (r["timestamp_leaks"] or r["tag_leaks"]
                             or r["max_consecutive_repeat"] >= 3 or gib >= 10) else ""
            out.append(f"| {r['adapter']} | {r['framework']} | {r['decode']} | "
                       f"{r['words']} | {r['ratio']:.2f} | {r['timestamp_leaks']} | "
                       f"{r['tag_leaks']} | {r['max_consecutive_repeat']} | {gib}{flag} | "
                       f"{r['top_sections']} | {r['vote_triples']} |")
        out.append("")

    # baseline vs antirep delta per (adapter, example)
    out += ["## baseline → antirep delta", "",
            "| example | adapter | Δmaxrep | Δts | Δtag | base→anti words |",
            "|---|---|---|---|---|---|"]
    keyed = {(r["example"], r["adapter"], r["decode"]): r for r in rows}
    for (ex, ad) in sorted({(r["example"], r["adapter"]) for r in rows}):
        b = keyed.get((ex, ad, "baseline"))
        a = keyed.get((ex, ad, "antirep"))
        if not (b and a):
            continue
        out.append(f"| {ex} | {ad} | {a['max_consecutive_repeat']-b['max_consecutive_repeat']} | "
                   f"{a['timestamp_leaks']-b['timestamp_leaks']} | "
                   f"{a['tag_leaks']-b['tag_leaks']} | "
                   f"{b['words']}→{a['words']} |")
    out.append("")

    out.append(FINDINGS.strip())
    out.append("")
    (run_dir / "COMPARISON.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {run_dir/'COMPARISON.md'} ({len(rows)} summaries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
