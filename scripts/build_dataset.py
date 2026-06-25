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
import hashlib
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz
from tqdm import tqdm

from docx_to_markdown import convert_docx
from utils.model_utils import context_window
from pdf_to_markdown import convert_pdf, make_converter
from preprocess_protocol import clean_protocol, split_front_matter
from utils.prompt_io import (
    build_user_message,
    load_summary_prompt,
    render_transcript_text,
    top_title_from_protocol,
)

# Source of truth for the system prompt (training targets + inference). Inlined
# copies in scripts/eval_io.py and scripts/infer_unsloth.py must be kept in sync.
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

TOP_TAG_RE = re.compile(r"<SD-TOP>(.*?)</SD>", re.DOTALL)
TOP_NUM_RE = re.compile(r"(?i)\b(?:TOP|Tagesordnungspunkt)\s*(\d+)")
# Anchor on the markdown heading ("## Zu TOP N"), NOT a bare "zu TOP N": matching
# mid-line stripped each heading's "## " marker into the slice boundary (heading lost
# its "##", and the next heading's "##" dangled as a trailing "##" on the previous
# section). Anchoring keeps the "## " with its section and ignores prose back-references.
PROT_TOP_RE = re.compile(r"(?im)^[ \t]*#{1,6}[ \t]*zu\s+TOP\s*(\d+)\b")
# Committee protocols make two ascending passes over the TOPs: a terse decision
# summary, then the substantive discussion. Split each pass on its own so the
# numbering stays monotonic, then merge by TOP.
PROT_BESCHLUSS_RE = re.compile(r"(?im)^##\s*Beschlüsse und Festlegungen")
PROT_BERATUNG_RE = re.compile(r"(?im)^##\s*Aus der Beratung")
PROT_ANLAGE_RE = re.compile(r"(?im)^##\s*Anlage")


def approx_tokens(text: str) -> int:
    """Cheap whitespace token estimate (avoids loading a tokenizer)."""
    return len(text.split())


def seq_tokens(tokenizer, messages: list[dict]) -> int:
    """Real token length of the full chat record (system+user+assistant), used to
    exclude examples that would not fit the model context window.

    ``return_dict=True`` so we get a list of ids; the bare ``tokenize=True`` form
    returns a BatchEncoding whose ``len`` is the number of keys (2), not tokens.
    """
    enc = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=False, return_dict=True)
    return len(enc["input_ids"])


def normalise_stem(stem: str) -> str:
    """Normalise a filename stem for pairing (drop role word, punctuation)."""
    s = stem.lower()
    for word in ("transkript", "protokoll"):
        s = s.replace(word, "")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_val_session(key: str, val_frac: float, seed: int) -> bool:
    """Deterministic, *pinned* train/val assignment for one session.

    Hashes the (seed, normalised key) so a session always lands on the same side
    regardless of how many other sessions exist — unlike a shuffle of the session
    list, where adding a new drop reshuffles the whole split. This keeps the val
    set stable across rebuilds (a fixed --seed alone does not, once the corpus
    changes)."""
    h = hashlib.md5(f"{seed}:{key}".encode("utf-8")).digest()
    return (int.from_bytes(h[:8], "big") % 10_000) < round(val_frac * 10_000)


def canon_key(stem: str) -> str:
    """Loose canonical form for holdout matching: normalise, then underscores -> spaces.

    Session keys keep a trailing ``_`` from the stripped role word (``ARD_1_Transkript``
    -> ``ard_1_``), while a bare manifest stem normalises to ``ard_1``; collapsing
    underscores to spaces makes both ``ard 1`` so they compare equal."""
    return re.sub(r"\s+", " ", normalise_stem(stem).replace("_", " ")).strip()


def read_holdout(manifest: Path | None, extra: list[str]) -> set[str]:
    """Canonical session keys to exclude from train+val (e.g. the eval test set).

    Reads the ``stem`` column of a TSV manifest (header required) plus any
    ``--holdout-stems``; both pass through ``canon_key`` to match internal keys."""
    stems: list[str] = list(extra)
    if manifest and manifest.exists():
        lines = manifest.read_text(encoding="utf-8").splitlines()
        if lines:
            header = lines[0].split("\t")
            idx = header.index("stem") if "stem" in header else 1
            for line in lines[1:]:
                cols = line.split("\t")
                if len(cols) > idx and cols[idx].strip():
                    stems.append(cols[idx].strip())
    return {canon_key(s) for s in stems}


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


# A "## Zu TOP N[:]" heading line, capturing any inline title remainder.
_TOP_HEAD_RE = re.compile(r"(?im)^#{1,6}[ \t]*Zu\s+TOP\s*(\d+)[ \t]*:?[ \t]*(.*)$")
# Paragraph that begins the discussion (a speaker/role/report opener), so the
# agenda title never swallows prose when it wraps across lines.
_TITLE_STOP_RE = re.compile(
    r"(?i)^(Der |Die |Das |Herr |Frau |Vorsitzend|Stellv|Bericht\b|Minister|"
    r"Staatssekret|Abgeordnet|Er |Sie |Es |Auf |Zunächst|Eingangs|Im Anschluss|"
    r"Einleitend|Anschließend)")


def _heading_title_body(section: str) -> tuple[str, str]:
    """Split one pass section into (title, body).

    The section opens with a ``## Zu TOP N[:]`` heading; its inline remainder plus
    any short wrapped continuation paragraphs (PDF line-breaks turned the agenda
    title into several paragraphs) form the title, and the rest is the body. A
    continuation paragraph joins the title only while it stays short, has no
    sentence-ending punctuation and does not look like the start of the
    discussion (``_TITLE_STOP_RE``)."""
    s = section.lstrip("\n")
    m = _TOP_HEAD_RE.match(s)
    if not m:
        return "", section.strip()
    rest = s[m.end():].lstrip("\n")
    title_parts = [m.group(2).strip()] if m.group(2).strip() else []
    paras = re.split(r"\n\s*\n", rest)
    body_start = 0
    if title_parts:  # only extend an existing inline title
        for i, p in enumerate(paras):
            pp = p.strip()
            if (i < 3 and pp and len(pp) <= 120 and not pp.startswith("#")
                    and not re.search(r"[.!?]$", pp) and not _TITLE_STOP_RE.match(pp)):
                title_parts.append(pp)
                body_start = i + 1
            else:
                break
    title = re.sub(r"\s+", " ", " ".join(title_parts)).strip(" :-–")
    body = "\n\n".join(paras[body_start:]).strip()
    return title, body


def _canonical_top(n: int, beschluss_sec: str, beratung_sec: str) -> str:
    """One canonical section per TOP: a single ``## Zu TOP N: <title>`` heading,
    then the Beschluss, then an ``Aus der Beratung`` separator and the discussion.

    Collapses the source's two-pass layout (a colon ``## Zu TOP N:`` Beschluss
    heading plus a ``## Zu TOP N <title>`` Beratung heading) into one, so every
    target shares the same shape regardless of whether the TOP carried a decision."""
    b_title, b_body = _heading_title_body(beschluss_sec) if beschluss_sec.strip() else ("", "")
    r_title, r_body = _heading_title_body(beratung_sec) if beratung_sec.strip() else ("", "")
    title = r_title or b_title  # the Beratung pass carries the agenda title
    parts = [f"## Zu TOP {n}: {title}".rstrip() if title else f"## Zu TOP {n}:"]
    if b_body:
        parts.append(b_body)
    if r_body:
        if b_body:
            parts.append("Aus der Beratung")
        parts.append(r_body)
    return "\n\n".join(parts).strip()


def split_protocol_by_top(text: str) -> dict[int, str]:
    """Map TOP number -> one canonical section (heading + Beschluss + discussion).

    Splits the ``Beschlüsse und Festlegungen`` and ``Aus der Beratung`` passes
    independently (each monotonic on its own) and merges them per TOP via
    ``_canonical_top``. Falls back to a single monotonic pass over the whole text
    when the section headings are absent (e.g. non-standard layout)."""
    beschluss, beratung = _section_bounds(text)
    if not beschluss and not beratung:
        return {n: _canonical_top(n, "", sec) for n, sec in _split_pass(text).items()}
    bm = _split_pass(beschluss)
    rm = _split_pass(beratung)
    return {n: _canonical_top(n, bm.get(n, ""), rm.get(n, ""))
            for n in sorted(set(bm) | set(rm))}


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
    p.add_argument("--protocol-dir", type=Path, default=Path("data/protocols/md"),
                   help="Directory of protocol files, .md/.pdf (default: data/protocols/md; "
                        "protocols are cleaned internally, so the raw md/ dir is fine)")
    p.add_argument("--out-dir", type=Path, default=Path("data/train"),
                   help="Output directory for train.jsonl / val.jsonl (default: data/train)")
    p.add_argument("--granularity", choices=("document", "per-top"), default="per-top",
                   help="Segmentation strategy (default: per-top)")
    p.add_argument("--include-untagged-as-document", action="store_true",
                   help="per-top only: sessions with no aligned TOPs (e.g. single-TOP budget "
                        "sittings B1 couldn't anchor) are SKIPPED by default and logged to "
                        "<out-dir>/untagged_sessions.json. Pass this to instead include each such "
                        "session as one whole-document record (still subject to --min-tgt-tokens).")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of sessions held out for validation (default: 0.1)")
    p.add_argument("--min-tgt-tokens", type=int, default=32,
                   help="Drop records whose target is shorter than this (default: 32)")
    p.add_argument("--match-threshold", type=float, default=90.0,
                   help="Min fuzzy score to accept a transcript/protocol pair (default: 90)")
    p.add_argument("--marker", default=r"(?i)zu\s+TOP\s*1\b",
                   help="Protocol body-start marker regex")
    p.add_argument("--base-model", default="google/gemma-4-31B-it",
                   help="Model whose tokenizer + context window decide token counts and the "
                        "length exclusion (default: google/gemma-4-31B-it)")
    p.add_argument("--max-seq-len", type=int, default=65536,
                   help="Max record length in real tokens; longer records are EXCLUDED (not "
                        "truncated) and recorded in --exclusions. Default: 65536 (65k cap). "
                        "Pass 0 to fall back to the base model's full context window.")
    p.add_argument("--val-max-seq-len", type=int, default=8192,
                   help="Cap on validation-record length in real tokens (default: 8192). A "
                        "val-assigned record longer than this (but within --max-seq-len) is "
                        "routed to train instead of dropped, keeping the eval forward pass cheap. "
                        "Pass 0 to disable the cap (val keeps everything up to --max-seq-len).")
    p.add_argument("--system-prompt-file", type=Path, default=None,
                   help="File with a custom system prompt (default: built-in German prompt)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the pinned per-session train/val hash (default: 42)")
    p.add_argument("--holdout-manifest", type=Path, default=None,
                   help="TSV with a 'stem' column whose sessions are EXCLUDED from train+val "
                        "(e.g. test/manifest.tsv, the held-out eval set)")
    p.add_argument("--holdout-stems", nargs="*", default=[],
                   help="Extra session stems to exclude from train+val (normalised internally)")
    p.add_argument("--exclusions", type=Path, default=None,
                   help="JSON {stem: [tops]} of per-TOP records to skip "
                        "(e.g. match_speakers.py exclusions.json)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing train.jsonl / val.jsonl")
    args = p.parse_args()

    excluded: dict[str, set] = {}
    if args.exclusions and args.exclusions.exists():
        raw = json.loads(args.exclusions.read_text(encoding="utf-8"))
        excluded = {k: set(v) for k, v in raw.items()}

    holdout = read_holdout(args.holdout_manifest, args.holdout_stems)
    if holdout:
        print(f"holdout (excluded from train+val): {len(holdout)} session(s): "
              f"{', '.join(sorted(holdout))}", file=sys.stderr)

    if not args.max_seq_len:  # None or 0 -> fall back to the model's full context window
        args.max_seq_len = context_window(args.base_model)
    print(f"max-seq-len: {args.max_seq_len} tokens (base {args.base_model})", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # length-based exclusions discovered during this build (merged into exclusions.json)
    length_excluded: dict[str, set] = {}

    for d in (args.transcript_dir, args.protocol_dir):
        if not d.is_dir():
            print(f"{d} is not a directory", file=sys.stderr)
            return 1

    system = (args.system_prompt_file.read_text(encoding="utf-8").strip()
              if args.system_prompt_file else load_summary_prompt())

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
    untagged: list[str] = []  # sessions with no aligned TOPs (logged; included only on flag)
    failures: list[tuple[str, str]] = []
    # per-TOP exclusion ledger for <out-dir>/exclusions_report.tsv:
    # (stem, top, reason, seq_tokens, disposition)
    exclusion_rows: list[tuple[str, str, str, str, str]] = []
    for tx_path, pr_path in tqdm(pairs, desc="build", unit="pair"):
        key = normalise_stem(tx_path.stem)
        if canon_key(tx_path.stem) in holdout:  # held-out eval session: never in train/val
            print(f"  holdout: skipping {tx_path.name}", file=sys.stderr)
            continue
        try:
            transcript = read_transcript(tx_path, converter=converter)
            protocol = read_protocol(pr_path, marker=args.marker, converter=converter)
        except Exception as exc:
            failures.append((tx_path.name, repr(exc)))
            print(f"\nERROR on {tx_path.name}: {exc!r}", file=sys.stderr)
            continue

        recs: list[dict] = []

        def consider(rec: dict, label) -> None:
            """Keep the record unless it exceeds the context window; over-length
            records are excluded (not truncated) and recorded for exclusions.json."""
            st = seq_tokens(tokenizer, rec["messages"])
            rec["meta"]["seq_tokens"] = st
            if st > args.max_seq_len:
                length_excluded.setdefault(key, set()).add(label)
                exclusion_rows.append(
                    (key, str(label), "length-excluded", str(st),
                     f"excluded (> max_seq_len {args.max_seq_len})"))
            else:
                recs.append(rec)

        if args.granularity == "per-top":
            tx_tops = split_transcript_by_top(transcript)
            pr_tops = split_protocol_by_top(protocol)
            common = sorted(set(tx_tops) & set(pr_tops))
            for n in common:
                if n in excluded.get(key, ()):  # unresolved speaker(s) -> drop
                    excluded_n += 1
                    exclusion_rows.append(
                        (key, str(n), "speaker-excluded", "",
                         "excluded (unresolved speaker; --exclusions)"))
                    continue
                if approx_tokens(pr_tops[n]) < args.min_tgt_tokens:
                    dropped += 1
                    exclusion_rows.append(
                        (key, str(n), "target-too-short", str(approx_tokens(pr_tops[n])),
                         f"excluded (target < min_tgt_tokens {args.min_tgt_tokens})"))
                    continue
                user = build_user_message(top_title_from_protocol(protocol, n),
                                          render_transcript_text(tx_tops[n]))
                consider(make_record(system, user, pr_tops[n],
                                     {"stem": key, "top": n, "strategy": "per-top"}), n)
            if not common:  # no aligned TOPs located in this session
                untagged.append(key)
                if args.include_untagged_as_document and \
                        approx_tokens(protocol) >= args.min_tgt_tokens:
                    print(f"  no aligned TOPs for {tx_path.name}; including as whole document",
                          file=sys.stderr)
                    consider(make_record(system, transcript, protocol,
                                         {"stem": key, "strategy": "document-fallback"}), "document")
                    exclusion_rows.append(
                        (key, "*", "no-aligned-TOPs", "",
                         "recovered as whole-document record (--include-untagged-as-document)"))
                else:
                    why = ("excluded by default" if not args.include_untagged_as_document
                           else "skipped: target too short")
                    print(f"  no aligned TOPs for {tx_path.name}; {why} "
                          f"(logged to untagged_sessions.json)", file=sys.stderr)
                    exclusion_rows.append(
                        (key, "*", "no-aligned-TOPs", "", f"excluded ({why})"))
            elif not recs:  # had aligned TOPs but all excluded/short -> contribute nothing
                print(f"  all aligned TOPs filtered for {tx_path.name}; skipped",
                      file=sys.stderr)
        else:
            if approx_tokens(protocol) >= args.min_tgt_tokens:
                consider(make_record(system, transcript, protocol,
                                     {"stem": key, "strategy": "document"}), "document")
        sessions.setdefault(key, []).extend(recs)

    sessions = {k: v for k, v in sessions.items() if v}
    if not sessions:
        print("no records produced", file=sys.stderr)
        return 2

    # Pinned per-session train/val split: each session is assigned by hashing its
    # own key, so the val set is stable across rebuilds and new drops never reshuffle
    # existing sessions across the split.
    keys = sorted(sessions)
    val_keys = {k for k in keys if is_val_session(k, args.val_frac, args.seed)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    for path in (train_path, val_path):
        if path.exists() and not args.overwrite:
            print(f"{path} exists; use --overwrite", file=sys.stderr)
            return 1

    # Validation length cap: a val-assigned record longer than --val-max-seq-len
    # (but within --max-seq-len) is routed to TRAIN rather than dropped, keeping the
    # eval forward pass cheap. This relaxes the by-session no-straddle invariant on
    # purpose: a val session's long TOPs may land in train while its short TOPs stay
    # in val (different TOPs/targets, so no target leakage).
    val_cap = args.val_max_seq_len
    n_train = n_val_recs = n_val_to_train = 0
    with train_path.open("w", encoding="utf-8") as ft, val_path.open("w", encoding="utf-8") as fv:
        for key in keys:
            is_val = key in val_keys
            for rec in sessions[key]:
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                if is_val and val_cap and rec["meta"]["seq_tokens"] > val_cap:
                    ft.write(line)  # too long for val -> train
                    n_train += 1
                    n_val_to_train += 1
                    exclusion_rows.append(
                        (key, str(rec["meta"].get("top", "document")), "val-overflow",
                         str(rec["meta"]["seq_tokens"]),
                         f"moved to train (> val_max_seq_len {val_cap})"))
                elif is_val:
                    fv.write(line)
                    n_val_recs += 1
                else:
                    ft.write(line)
                    n_train += 1

    len_excluded_n = sum(len(v) for v in length_excluded.values())
    tgt_tok = [r["meta"]["tgt_tokens"] for v in sessions.values() for r in v]
    src_tok = [r["meta"]["src_tokens"] for v in sessions.values() for r in v]
    seq_tok = [r["meta"]["seq_tokens"] for v in sessions.values() for r in v]
    print("\n=== dataset summary ===", file=sys.stderr)
    print(f"sessions: {len(keys)} ({len(val_keys)} val)", file=sys.stderr)
    print(f"records:  {n_train} train, {n_val_recs} val "
          f"(dropped {dropped} short, {excluded_n} speaker-excluded, "
          f"{len_excluded_n} length-excluded > {args.max_seq_len} tokens)", file=sys.stderr)
    if val_cap:
        print(f"val cap:  {val_cap} tokens; {n_val_to_train} val record(s) over the cap "
              f"routed to train", file=sys.stderr)
    if src_tok:
        print(f"src tokens (approx): min {min(src_tok)} / "
              f"median {sorted(src_tok)[len(src_tok)//2]} / max {max(src_tok)}", file=sys.stderr)
        print(f"tgt tokens (approx): min {min(tgt_tok)} / "
              f"median {sorted(tgt_tok)[len(tgt_tok)//2]} / max {max(tgt_tok)}", file=sys.stderr)
        print(f"seq tokens (real):   min {min(seq_tok)} / "
              f"median {sorted(seq_tok)[len(seq_tok)//2]} / max {max(seq_tok)}", file=sys.stderr)
    print(f"wrote {train_path} and {val_path}", file=sys.stderr)

    if untagged:
        uniq = sorted(set(untagged))
        (args.out_dir / "untagged_sessions.json").write_text(
            json.dumps(uniq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        action = ("included as whole documents" if args.include_untagged_as_document
                  else "EXCLUDED (pass --include-untagged-as-document to keep)")
        print(f"untagged sessions (no aligned TOPs): {len(uniq)} {action}; "
              f"logged to {args.out_dir / 'untagged_sessions.json'}", file=sys.stderr)

    # Per-TOP exclusion report: every dropped/rerouted item with its reason, so the
    # excluded material can be reviewed before deciding what to recover into with-docs.
    if exclusion_rows:
        report_path = args.out_dir / "exclusions_report.tsv"
        order = {"speaker-excluded": 0, "target-too-short": 1, "length-excluded": 2,
                 "no-aligned-TOPs": 3, "val-overflow": 4}
        rows = sorted(exclusion_rows, key=lambda r: (order.get(r[2], 9), r[0], r[1]))
        lines = ["stem\ttop\treason\tseq_tokens\tdisposition"]
        lines += ["\t".join(r) for r in rows]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts: dict[str, int] = {}
        for r in rows:
            counts[r[2]] = counts.get(r[2], 0) + 1
        tally = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"exclusion report: {len(rows)} row(s) ({tally}) -> {report_path}",
              file=sys.stderr)

    # Merge length exclusions into the exclusions file (union with speaker-based
    # exclusions). Records over the context window are excluded, not truncated.
    if length_excluded:
        merged: dict[str, set] = {k: set(v) for k, v in excluded.items()}
        for k, v in length_excluded.items():
            merged.setdefault(k, set()).update(v)
        excl_path = args.exclusions or (args.out_dir / "exclusions.json")
        excl_path.write_text(
            json.dumps({k: sorted(v, key=lambda x: (isinstance(x, str), x))
                        for k, v in sorted(merged.items())},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"merged {len_excluded_n} length exclusion(s) into {excl_path}", file=sys.stderr)

    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
