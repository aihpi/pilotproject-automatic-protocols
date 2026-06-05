#!/usr/bin/env python3
"""Build train/val JSONL from paired transcripts and protocols.

Pairs each transcript (``… Transkript.docx`` / ``.md``) with its protocol
(``… Protokoll.pdf`` / ``.md``) by fuzzy-matching the normalised filename stem
(robust to quirks like ``AWAEK``↔``AWEK`` and a missing period). Each pair
becomes one or more TRL chat records:

    {"messages": [{"role": "system", ...},
                  {"role": "user", "content": <transcript>},
                  {"role": "assistant", "content": <protocol>}],
     "meta": {...}}

Granularity ``per-top`` (default) splits both sides on agenda item (transcript
``<SD-TOP>`` numbers, protocol ``Zu TOP N`` headings) and emits one record per
aligned item — many more, shorter examples. ``document`` emits one record per
session. The train/val split is by session, so a session's items never straddle
the split. Runs on CPU; no GPU needed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz
from tqdm import tqdm

from docx_to_markdown import convert_docx
from pdf_to_markdown import convert_pdf, make_converter
from preprocess_protocol import clean_protocol, split_front_matter

DEFAULT_SYSTEM_PROMPT = (
    "Du bist Protokollführer/in eines Ausschusses des Landtags Brandenburg. "
    "Wandle das wörtliche Transkript einer Sitzung in ein formelles "
    "Ausschussprotokoll im amtlichen Stil um. Nutze die Sprecher- und "
    "Tagesordnungs-Markierungen, fasse die Beratung sachlich zusammen und gib "
    "Beschlüsse und Abstimmungsergebnisse präzise wieder."
)

TOP_TAG_RE = re.compile(r"<SD-TOP>(.*?)</SD>", re.DOTALL)
TOP_NUM_RE = re.compile(r"(?i)\b(?:TOP|Tagesordnungspunkt)\s*(\d+)")
PROT_TOP_RE = re.compile(r"(?i)\bzu\s+TOP\s*(\d+)\b")
# Committee protocols make two ascending passes over the TOPs: a terse decision
# summary, then the substantive discussion. Split each pass on its own so the
# numbering stays monotonic, then merge by TOP.
PROT_BESCHLUSS_RE = re.compile(r"(?im)^##\s*Beschlüsse und Festlegungen")
PROT_BERATUNG_RE = re.compile(r"(?im)^##\s*Aus der Beratung")
PROT_ANLAGE_RE = re.compile(r"(?im)^##\s*Anlage")


def approx_tokens(text: str) -> int:
    """Cheap whitespace token estimate (avoids loading a tokenizer)."""
    return len(text.split())


def normalise_stem(stem: str) -> str:
    """Normalise a filename stem for pairing (drop role word, punctuation)."""
    s = stem.lower()
    for word in ("transkript", "protokoll"):
        s = s.replace(word, "")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def read_transcript(path: Path, *, converter=None) -> str:
    if path.suffix.lower() == ".docx":
        return convert_docx(path, converter, keep_tags=True)
    _, body = split_front_matter(path.read_text(encoding="utf-8"))
    return body.strip()


def read_protocol(path: Path, *, marker: str, converter=None) -> str:
    if path.suffix.lower() == ".pdf":
        text = convert_pdf(path, converter)
    else:
        text = path.read_text(encoding="utf-8")
    # Cover off, attachments and page noise stripped. Idempotent on already-clean
    # md_clean files; also handles raw .md / freshly converted .pdf. (``marker``
    # is unused for committee protocols, accepted for backward compatibility.)
    _, body, _ = clean_protocol(text)
    return body.strip()


def _monotonic_sections(boundaries: list[tuple[int, int]], text: str) -> dict[int, str]:
    """Given (number, start) boundaries in document order, slice into sections.

    Only accepts strictly increasing numbers (ignores back-references like a
    later "zu TOP 1" mention), which matches the monotonic agenda numbering.
    """
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
    boundaries: list[tuple[int, int]] = []
    for m in TOP_TAG_RE.finditer(text):
        num_m = TOP_NUM_RE.search(m.group(1))
        if num_m:
            boundaries.append((int(num_m.group(1)), m.start()))
    return _monotonic_sections(boundaries, text)


def _section_bounds(text: str) -> tuple[str, str]:
    """Slice a cleaned protocol into its (Beschlüsse, Aus der Beratung) passes."""
    b = PROT_BESCHLUSS_RE.search(text)
    r = PROT_BERATUNG_RE.search(text)
    a = PROT_ANLAGE_RE.search(text)
    end = a.start() if a else len(text)
    if r:
        beschluss = text[b.start():r.start()] if b else text[:r.start()]
        beratung = text[r.start():end]
    else:
        beschluss = text[b.start():end] if b else ""
        beratung = ""
    return beschluss, beratung


def _split_pass(text: str) -> dict[int, str]:
    boundaries = [(int(m.group(1)), m.start()) for m in PROT_TOP_RE.finditer(text)]
    return _monotonic_sections(boundaries, text)


def split_protocol_by_top(text: str) -> dict[int, str]:
    """Map TOP number -> decision summary + discussion for that agenda item.

    Splits the ``Beschlüsse und Festlegungen`` and ``Aus der Beratung`` passes
    independently (each monotonic on its own) and merges them per TOP. Falls back
    to a single monotonic pass over the whole text when the section headings are
    absent (e.g. non-standard layout)."""
    beschluss, beratung = _section_bounds(text)
    if not beschluss and not beratung:
        return _split_pass(text)
    bm = _split_pass(beschluss)
    rm = _split_pass(beratung)
    out: dict[int, str] = {}
    for n in sorted(set(bm) | set(rm)):
        out[n] = "\n\n".join(s for s in (bm.get(n, "").strip(),
                                          rm.get(n, "").strip()) if s)
    return out


def make_record(system: str, user: str, assistant: str, meta: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {**meta,
                 "src_tokens": approx_tokens(user),
                 "tgt_tokens": approx_tokens(assistant)},
    }


def pair_files(transcripts: list[Path], protocols: list[Path],
               threshold: float) -> list[tuple[Path, Path]]:
    """Fuzzy-match transcripts to protocols by normalised filename stem."""
    pairs: list[tuple[Path, Path]] = []
    used: set[Path] = set()
    for tx in transcripts:
        tx_key = normalise_stem(tx.stem)
        best, best_score = None, 0.0
        for pr in protocols:
            if pr in used:
                continue
            score = fuzz.token_sort_ratio(tx_key, normalise_stem(pr.stem))
            if score > best_score:
                best, best_score = pr, score
        if best is not None and best_score >= threshold:
            pairs.append((tx, best))
            used.add(best)
            print(f"  pair: {tx.name!r} <-> {best.name!r} (score {best_score:.0f})",
                  file=sys.stderr)
        else:
            print(f"  UNMATCHED transcript: {tx.name!r} "
                  f"(best {best.name if best else '-'} @ {best_score:.0f})", file=sys.stderr)
    for pr in protocols:
        if pr not in used:
            print(f"  UNMATCHED protocol:   {pr.name!r}", file=sys.stderr)
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts/md"),
                   help="Directory of transcript files, .md/.docx (default: data/transcripts/md)")
    p.add_argument("--protocol-dir", type=Path, default=Path("data/protocols/md_clean"),
                   help="Directory of protocol files, .md/.pdf (default: data/protocols/md_clean)")
    p.add_argument("--out-dir", type=Path, default=Path("data/train"),
                   help="Output directory for train.jsonl / val.jsonl (default: data/train)")
    p.add_argument("--granularity", choices=("document", "per-top"), default="per-top",
                   help="Segmentation strategy (default: per-top)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of sessions held out for validation (default: 0.1)")
    p.add_argument("--min-tgt-tokens", type=int, default=32,
                   help="Drop records whose target is shorter than this (default: 32)")
    p.add_argument("--match-threshold", type=float, default=90.0,
                   help="Min fuzzy score to accept a transcript/protocol pair (default: 90)")
    p.add_argument("--marker", default=r"(?i)zu\s+TOP\s*1\b",
                   help="Protocol body-start marker regex")
    p.add_argument("--system-prompt-file", type=Path, default=None,
                   help="File with a custom system prompt (default: built-in German prompt)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the session-level train/val shuffle (default: 42)")
    p.add_argument("--exclusions", type=Path, default=None,
                   help="JSON {stem: [tops]} of per-TOP records to skip "
                        "(e.g. match_speakers.py exclusions.json)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing train.jsonl / val.jsonl")
    args = p.parse_args()

    excluded: dict[str, set[int]] = {}
    if args.exclusions:
        raw = json.loads(args.exclusions.read_text(encoding="utf-8"))
        excluded = {k: set(v) for k, v in raw.items()}

    for d in (args.transcript_dir, args.protocol_dir):
        if not d.is_dir():
            print(f"{d} is not a directory", file=sys.stderr)
            return 1

    system = (args.system_prompt_file.read_text(encoding="utf-8").strip()
              if args.system_prompt_file else DEFAULT_SYSTEM_PROMPT)

    transcripts = sorted(f for f in args.transcript_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in (".docx", ".md"))
    protocols = sorted(f for f in args.protocol_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in (".pdf", ".md"))
    print(f"scanning {len(transcripts)} transcript(s) in {args.transcript_dir} and "
          f"{len(protocols)} protocol(s) in {args.protocol_dir}", file=sys.stderr)
    pairs = pair_files(transcripts, protocols, args.match_threshold)
    if not pairs:
        print("no transcript/protocol pairs found", file=sys.stderr)
        return 1

    # One shared Docling converter for any raw PDF protocols / DOCX transcripts.
    converter = None
    if any(p.suffix.lower() == ".pdf" for _, p in pairs) or \
            any(t.suffix.lower() == ".docx" for t, _ in pairs):
        converter = make_converter()

    # session_key -> list[record]
    sessions: dict[str, list[dict]] = {}
    dropped = 0
    excluded_n = 0
    failures: list[tuple[str, str]] = []
    for tx_path, pr_path in tqdm(pairs, desc="build", unit="pair"):
        key = normalise_stem(tx_path.stem)
        try:
            transcript = read_transcript(tx_path, converter=converter)
            protocol = read_protocol(pr_path, marker=args.marker, converter=converter)
        except Exception as exc:
            failures.append((tx_path.name, repr(exc)))
            print(f"\nERROR on {tx_path.name}: {exc!r}", file=sys.stderr)
            continue

        recs: list[dict] = []
        if args.granularity == "per-top":
            tx_tops = split_transcript_by_top(transcript)
            pr_tops = split_protocol_by_top(protocol)
            common = sorted(set(tx_tops) & set(pr_tops))
            for n in common:
                if n in excluded.get(key, ()):  # unresolved speaker(s) -> drop
                    excluded_n += 1
                    continue
                if approx_tokens(pr_tops[n]) < args.min_tgt_tokens:
                    dropped += 1
                    continue
                recs.append(make_record(system, tx_tops[n], pr_tops[n],
                                        {"stem": key, "top": n, "strategy": "per-top"}))
            if not common:  # alignment genuinely failed -> whole-document fallback
                print(f"  no aligned TOPs for {tx_path.name}; document fallback",
                      file=sys.stderr)
                recs.append(make_record(system, transcript, protocol,
                                        {"stem": key, "strategy": "document-fallback"}))
            elif not recs:  # had aligned TOPs but all excluded/short -> contribute nothing
                print(f"  all aligned TOPs filtered for {tx_path.name}; skipped",
                      file=sys.stderr)
        else:
            if approx_tokens(protocol) >= args.min_tgt_tokens:
                recs.append(make_record(system, transcript, protocol,
                                        {"stem": key, "strategy": "document"}))
        sessions.setdefault(key, []).extend(recs)

    sessions = {k: v for k, v in sessions.items() if v}
    if not sessions:
        print("no records produced", file=sys.stderr)
        return 2

    keys = sorted(sessions)
    random.Random(args.seed).shuffle(keys)
    n_val = max(1, round(len(keys) * args.val_frac)) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    for path in (train_path, val_path):
        if path.exists() and not args.overwrite:
            print(f"{path} exists; use --overwrite", file=sys.stderr)
            return 1

    n_train = n_val_recs = 0
    with train_path.open("w", encoding="utf-8") as ft, val_path.open("w", encoding="utf-8") as fv:
        for key in keys:
            sink = fv if key in val_keys else ft
            for rec in sessions[key]:
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if key in val_keys:
                    n_val_recs += 1
                else:
                    n_train += 1

    tgt_tok = [r["meta"]["tgt_tokens"] for v in sessions.values() for r in v]
    src_tok = [r["meta"]["src_tokens"] for v in sessions.values() for r in v]
    print("\n=== dataset summary ===", file=sys.stderr)
    print(f"sessions: {len(keys)} ({len(val_keys)} val)", file=sys.stderr)
    print(f"records:  {n_train} train, {n_val_recs} val "
          f"(dropped {dropped} short, {excluded_n} excluded)", file=sys.stderr)
    if src_tok:
        print(f"src tokens (approx): min {min(src_tok)} / "
              f"median {sorted(src_tok)[len(src_tok)//2]} / max {max(src_tok)}", file=sys.stderr)
        print(f"tgt tokens (approx): min {min(tgt_tok)} / "
              f"median {sorted(tgt_tok)[len(tgt_tok)//2]} / max {max(tgt_tok)}", file=sys.stderr)
    print(f"wrote {train_path} and {val_path}", file=sys.stderr)

    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
