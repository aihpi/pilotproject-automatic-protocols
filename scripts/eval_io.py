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

# --- system prompt -----------------------------------------------------------
# Inlined copy of build_dataset.DEFAULT_SYSTEM_PROMPT (the source of truth) —
# build_dataset pulls rapidfuzz/docling and can't be imported from the alt venvs.
# Keep IN SYNC with scripts/build_dataset.py when the prompt changes.
DEFAULT_SYSTEM_PROMPT = """Du bist Protokollführer/in eines Ausschusses. Wandle das wörtliche Sitzungstranskript in ein formelles Ausschussprotokoll im amtlichen Stil um.

Sprache und Stil:
- Schreibe ausschließlich auf Deutsch in korrektem, sachlichem Verwaltungsdeutsch.
- Gib Wortbeiträge in indirekter Rede (Konjunktiv I) und in der dritten Person wieder (z. B. „Er betont, dass …“, „Sie verweist darauf, dass …“).
- Nenne Sprecher/innen mit Name und Rolle/Fraktion, z. B. „Kristy Augustin (CDU)“, „Steffen Freiberg (Minister für Bildung, Jugend und Sport)“.

Formatierung:
- Gliedere nach Tagesordnungspunkten mit genau EINER Überschrift „## Zu TOP N:“ je Punkt (Nummer aus den <SD-TOP>-Markierungen).
- Formuliere Beschlüsse als „Der [Gremium] beschließt einstimmig/mehrheitlich (Ja : Nein : Enthaltungen) …“ und gib Abstimmungsergebnisse stets als konkretes Tripel (Ja : Nein : Enthaltungen) bzw. als „einstimmig“/„mehrheitlich“ an — niemals als leeren Platzhalter.
- Trenne, sofern vorhanden, Beschlüsse/Festlegungen von der Zusammenfassung der Beratung („Aus der Beratung“).

Umgang mit dem Rohmaterial (Transkript):
- Das Transkript ist eine automatische Verschriftlichung (ASR) mit Sprecher-Diarisierung; es enthält technische Markierungen und Erkennungsfehler, die NICHT ins Protokoll gehören.
- Übernimm KEINE Zeitstempel (z. B. „[00:12:34]“ oder „(00:00:00.000 --> …)“) und erzeuge auch keine eigenen Zeit- oder „Sitzungsbeginn“-Angaben.
- Entnimm die TOP-Nummer den <SD-TOP>-Markierungen, übernimm die Markierungen und Tags selbst (z. B. „<SD-SPK>“, „<SD-TOP>“, „SPEAKER_03“) aber nicht in den Text.
- Ignoriere offensichtliche Transkriptionsfehler und sinnlose Wiederholungen (z. B. mehrfach hintereinander „Vielen Dank.“); wiederhole sie nicht und werte sie nicht als Inhalt.

Inhaltliche Treue:
- Fasse ausschließlich zusammen, was tatsächlich gesagt wurde. Füge keine Inhalte, Wertungen oder Fakten hinzu, die nicht im Transkript stehen, und verändere oder verfälsche keine Aussagen (auch keine Namen oder Zahlen).
- Im Zweifel knapper und näher am Wortlaut bleiben."""

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
# antirep adds the handoff's recommended repetition_penalty + no_repeat_ngram.
# Only these two knobs differ, to isolate the decoding fix.
DECODE_PRESETS: dict[str, dict] = {
    "baseline": dict(temperature=0.3, top_p=0.9, max_new_tokens=4096,
                     repetition_penalty=1.0, no_repeat_ngram_size=0, min_new_tokens=0),
    "antirep": dict(temperature=0.3, top_p=0.9, max_new_tokens=4096,
                    repetition_penalty=1.3, no_repeat_ngram_size=3, min_new_tokens=0),
}


# --- example discovery & output routing --------------------------------------
@dataclass
class Example:
    name: str
    transcript_path: Path
    summaries_dir: Path


def discover_examples(test_dir: Path) -> list[Example]:
    out: list[Example] = []
    for sub in (p for p in test_dir.iterdir() if p.is_dir()):
        tx = sorted(sub.glob("*_Transkript.md"))
        if not tx:
            continue
        out.append(Example(sub.name, tx[0], sub / "summaries"))
    # shortest transcript first: cheap examples produce outputs early, so a job
    # that hits its time limit still leaves the small results behind.
    out.sort(key=lambda e: e.transcript_path.stat().st_size)
    return out


def result_path(summaries_dir: Path, adapter_id: str, framework: str,
                granularity: str, decode: str) -> Path:
    return summaries_dir / f"{adapter_id}__{framework}__{granularity}__{decode}.md"


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
        return gen(transcript)
    tops = split_transcript_by_top(transcript)
    if not tops:
        log("  no numbered TOPs found; whole-document fallback")
        return gen(transcript)
    sections = []
    for n in sorted(tops):
        log(f"  TOP {n} ({len(tops[n])} chars)")
        sections.append(f"## Zu TOP {n}\n\n{gen(tops[n])}")
    return "\n\n".join(sections)


def run_test_set(generate_fn: Callable[[str, str, dict], str], *,
                 adapter_id: str, framework: str, base_model: str,
                 granularity: str, test_dir: Path, decodes: list[str],
                 system: str = DEFAULT_SYSTEM_PROMPT, overwrite: bool = False,
                 only: str | None = None, log=print) -> list[Path]:
    """For each example x decode preset: summarise and write to summaries/.

    ``generate_fn(system, user_text, decode_kwargs) -> str`` is supplied by the
    backend (PEFT / Unsloth / Keras). Skips outputs that already exist unless
    ``overwrite``. Returns the list of written paths.
    """
    examples = discover_examples(test_dir)
    if only:
        examples = [e for e in examples if only in e.name]
    if not examples:
        raise SystemExit(f"no examples with *_Transkript.md under {test_dir}"
                         + (f" matching {only!r}" if only else ""))
    written: list[Path] = []
    for ex in examples:
        transcript = strip_front_matter(ex.transcript_path.read_text(encoding="utf-8"))
        for decode in decodes:
            kwargs = DECODE_PRESETS[decode]
            out = result_path(ex.summaries_dir, adapter_id, framework, granularity, decode)
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
