#!/usr/bin/env python3
"""Shared, dependency-light helpers for the LoRA evaluation harness.

Imported by ``eval_lora.py`` (and usable from any framework venv: Unsloth,
Keras, PEFT) — so it must stay **stdlib-only**. The transcript-splitting logic
and system prompt are inlined copies of ``build_dataset`` (which pulls in
rapidfuzz/docling and cannot be imported from the alt-framework venvs).

The harness standardises preprocessing across frameworks so that the only
variables in the comparison are (a) the adapter/framework and (b) the decoding
preset. ``run_test_set`` walks ``data/test/<example>/``, summarises each example
once per decode preset, and writes the result to that example's ``summaries/``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Input contract lives in prompt_io (stdlib-only, so importable from the
# alt-framework venvs too). Single source of truth for the prompt + the
# deployment-format user message; no more inlined drift.
from prompt_io import build_user_message, load_summary_prompt, render_transcript_text

# --- system prompt -----------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = load_summary_prompt()

# --- transcript TOP splitting (inlined from build_dataset, 2-pass monotonic) --
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
TOP_TAG_RE = re.compile(r"<SD-TOP>(.*?)</SD>", re.DOTALL)
TOP_NUM_RE = re.compile(r"(?i)\b(?:TOP|Tagesordnungspunkt)\s*(\d+)")


def strip_front_matter(text: str) -> str:
    return _FRONT_MATTER_RE.sub("", text).strip()


def _monotonic_sections(boundaries: list[tuple[int, int]], text: str) -> dict[int, str]:
    """Slice (number, start) boundaries (document order) into sections, keeping
    only strictly increasing numbers (ignores back-references)."""
    kept: list[tuple[int, int]] = []
    last = 0
    for num, start in boundaries:
        if num > last:
            kept.append((num, start))
            last = num
    out: dict[int, str] = {}
    for i, (num, start) in enumerate(kept):
        end = kept[i + 1][1] if i + 1 < len(kept) else len(text)
        out[num] = text[start:end].strip()
    return out


def split_transcript_by_top(text: str) -> dict[int, str]:
    """Map TOP number -> transcript segment at each ``<SD-TOP>`` marker."""
    boundaries: list[tuple[int, int]] = []
    for m in TOP_TAG_RE.finditer(text):
        num_m = TOP_NUM_RE.search(m.group(1))
        if num_m:
            boundaries.append((int(num_m.group(1)), m.start()))
    return _monotonic_sections(boundaries, text)


# --- decoding presets ---------------------------------------------------------
# baseline reproduces the current production defaults (no repetition control);
# antirep adds a GENTLE repetition_penalty only. The earlier aggressive preset
# (repetition_penalty=1.3 + no_repeat_ngram_size=3) over-suppressed: forbidding
# every repeated 3-gram kills legitimate German phrases ("Der Ausschuss",
# "Ja : Nein : …") and the model collapsed into character-salad that the
# ts/tag/maxrep metrics miss. So: mild penalty (1.15), NO n-gram block. This nudges
# away from hard loops without starving the model of normal repetition.
# Only these two knobs differ from baseline, to isolate the decoding effect.
# max_new_tokens is the per-TOP OUTPUT budget (per-top granularity → a full protocol
# can be many × this). Set ABOVE the longest per-TOP training target (4761 tokens in
# data/train_*_cap65k) so a legitimate long section is never clipped: 4761 → 6144
# (next 1024 above + margin). Only ever *reached* when a model fails to emit EOS
# (a degenerate adapter), so raising it is free for well-behaved adapters.
DECODE_PRESETS: dict[str, dict] = {
    "baseline": dict(temperature=0.3, top_p=0.9, max_new_tokens=6144,
                     repetition_penalty=1.0, no_repeat_ngram_size=0, min_new_tokens=0),
    "antirep": dict(temperature=0.3, top_p=0.9, max_new_tokens=6144,
                    repetition_penalty=1.15, no_repeat_ngram_size=0, min_new_tokens=0),
}


# --- example discovery & output routing --------------------------------------
@dataclass
class Example:
    name: str
    transcript_path: Path
    gold_path: Path | None


def discover_examples(examples_dir: Path) -> list[Example]:
    """Find evaluation examples under ``examples_dir`` (one folder per example,
    each with a ``*_Transkript.md`` input and optional ``*_Protokoll.md`` gold)."""
    out: list[Example] = []
    for sub in (p for p in examples_dir.iterdir() if p.is_dir()):
        tx = sorted(sub.glob("*_Transkript.md"))
        if not tx:
            continue
        gold = sorted(sub.glob("*_Protokoll.md"))
        out.append(Example(sub.name, tx[0], gold[0] if gold else None))
    # shortest transcript first: cheap examples produce outputs early, so a job
    # that hits its time limit still leaves the small results behind.
    out.sort(key=lambda e: e.transcript_path.stat().st_size)
    return out


def result_path(run_dir: Path, example_name: str, adapter_id: str, framework: str,
                granularity: str, decode: str) -> Path:
    # one run folder per eval-matrix run (like results/<ts>/): outputs grouped by
    # example, named <adapter>__<fw>__<granularity>__<decode>.md.
    return run_dir / example_name / f"{adapter_id}__{framework}__{granularity}__{decode}.md"


def write_summary(out_path: Path, *, meta: dict, body: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["---"] + [f"{k}: {v}" for k, v in meta.items()] + ["---", ""]
    out_path.write_text("\n".join(header) + body + "\n", encoding="utf-8")


# --- summarisation orchestration ---------------------------------------------
def summarise(transcript: str, granularity: str,
              gen: Callable[[str], str], log=print) -> str:
    """gen(user_text) -> model output. Per-top splits on <SD-TOP>; document does
    one pass. Falls back to whole-document when no numbered TOPs are present."""
    if granularity == "document":
        return gen(build_user_message("Gesamtes Protokoll", render_transcript_text(transcript)))
    tops = split_transcript_by_top(transcript)
    if not tops:
        log("  no numbered TOPs found; whole-document fallback")
        return gen(build_user_message("Gesamtes Protokoll", render_transcript_text(transcript)))
    sections = []
    for n in sorted(tops):
        log(f"  TOP {n} ({len(tops[n])} chars)")
        # Same deployment-format input as training (prompt_io.build_user_message);
        # serve has no gold protocol, so the title is the "TOP {n}" fallback.
        user = build_user_message(f"TOP {n}", render_transcript_text(tops[n]))
        sections.append(f"## Zu TOP {n}\n\n{gen(user)}")
    return "\n\n".join(sections)


def run_test_set(generate_fn: Callable[[str, str, dict], str], *,
                 adapter_id: str, framework: str, base_model: str,
                 granularity: str, examples_dir: Path, run_dir: Path,
                 decodes: list[str],
                 system: str = DEFAULT_SYSTEM_PROMPT, overwrite: bool = False,
                 only: str | None = None, log=print) -> list[Path]:
    """For each example x decode preset: summarise and write under ``run_dir``.

    Inputs come from ``examples_dir/<example>/`` (stable), outputs go to
    ``run_dir/<example>/`` (one timestamped folder per eval-matrix run).
    ``generate_fn(system, user_text, decode_kwargs) -> str`` is supplied by the
    backend (PEFT / Unsloth / Keras). Skips outputs that already exist unless
    ``overwrite``. Returns the list of written paths.
    """
    examples = discover_examples(examples_dir)
    if only:
        examples = [e for e in examples if only in e.name]
    if not examples:
        raise SystemExit(f"no examples with *_Transkript.md under {examples_dir}"
                         + (f" matching {only!r}" if only else ""))
    written: list[Path] = []
    for ex in examples:
        transcript = strip_front_matter(ex.transcript_path.read_text(encoding="utf-8"))
        for decode in decodes:
            kwargs = DECODE_PRESETS[decode]
            out = result_path(run_dir, ex.name, adapter_id, framework, granularity, decode)
            if out.exists() and not overwrite:
                log(f"skip (exists) {out.name}")
                continue
            log(f"=== {ex.name} | {decode} -> {out.name} ===")
            body = summarise(transcript, granularity,
                             lambda u: generate_fn(system, u, kwargs), log=log)
            write_summary(out, meta={
                "source": ex.transcript_path.name,
                "example": ex.name,
                "kind": "protocol-summary",
                "adapter_id": adapter_id,
                "framework": framework,
                "base_model": base_model,
                "granularity": granularity,
                "decode": decode,
                "decode_params": {k: kwargs[k] for k in
                                  ("temperature", "top_p", "repetition_penalty",
                                   "no_repeat_ngram_size", "max_new_tokens")},
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, body=body)
            written.append(out)
    return written


# --- degeneration metrics (used by the Phase-6 comparison report) ------------
# transcript timestamps leak as either raw "[HH:MM:SS -->" or the model's own
# learned "(HH:MM:SS.mmm --> ...)" zero/paren form — catch both.
_TS_LINE_RE = re.compile(r"[\[(]\d{1,2}:\d{2}:\d{2}")
_TAG_RE = re.compile(r"<SD-[A-Z]+>")
_TOP_HEAD_RE = re.compile(r"(?im)^##\s*Zu\s+TOP\s*\d+")
# vote triple "Ja : Nein : Enthaltungen", e.g. "12 : 3 : 0" — require spaces
# around the colons so HH:MM:SS timestamps (no spaces) don't false-positive.
_VOTE_RE = re.compile(r"\d{1,3}\s+:\s+\d{1,3}\s+:\s+\d{1,3}")


def _is_gibberish_token(t: str) -> bool:
    """Heuristic for character-salad tokens (e.g. 'uuuuuu', '=====', 'ZUTOPNSJLV').

    Catches the over-aggressive-decoding failure mode that ts/tag/maxrep miss: a
    single char repeated >=4x, a run of symbols, or an implausibly long token."""
    if len(t) > 30:
        return True
    if len(t) >= 4 and len(set(t)) <= 2:           # 'uuuuu', 'aaaa', '===='
        return True
    alnum = sum(c.isalnum() for c in t)
    if len(t) >= 4 and alnum < 0.4 * len(t):        # mostly punctuation/symbols
        return True
    return False


def degeneration_metrics(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines()]
    nonempty = [ln for ln in lines if ln]
    max_run, run = 1, 1
    for i in range(1, len(nonempty)):
        run = run + 1 if nonempty[i] == nonempty[i - 1] else 1
        max_run = max(max_run, run)
    toks = text.split()
    gib = sum(_is_gibberish_token(t) for t in toks)
    return {
        "chars": len(text),
        "words": len(toks),
        "lines": len(lines),
        "timestamp_leaks": len(_TS_LINE_RE.findall(text)),
        "tag_leaks": len(_TAG_RE.findall(text)),
        "max_consecutive_repeat": max_run if nonempty else 0,
        "top_sections": len(_TOP_HEAD_RE.findall(text)),
        "vote_triples": len(_VOTE_RE.findall(text)),
        # share of character-salad tokens — catches over-aggressive-decoding collapse
        "gibberish_pct": round(100 * gib / max(1, len(toks)), 1),
    }
