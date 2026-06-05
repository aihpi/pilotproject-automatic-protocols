#!/usr/bin/env python3
"""Match speakers across a diarised transcript and its protocol, then anonymise both.

A diarised transcript carries pyannote voice-cluster labels (``SPEAKER_00`` …) that
have no relation to the real names in the protocol (``## Dr. Dietmar Woidke
(Ministerpräsident):``). For LoRA training we want **one consistent generic tag per
person in both files**, so the model never has to bridge ``SPEAKER_00`` → a real name.

This script maps each pyannote label to a canonical protocol speaker using a strict
**priority cascade** (a label resolved by a higher method is never overridden):

1. **Explicit chair naming** — the chair announces the next speaker inside their own
   turn ("… darf ich Herrn Ministerpräsidenten Dr. *Woidke* bitten", "Ich sehe Herrn
   Abgeordneten *Ossowski* am Mikrofon"). The label of the turn *immediately following*
   such a cue is voted to that surname (majority vote per label).
2. **Content matching** — for still-unmapped labels, fuzzy-match the label's concatenated
   utterances against each protocol speaker's concatenated text (rapidfuzz).
3. **Rednerliste / sequence** — for whatever remains, align the chronological order of
   transcript turns with the order of protocol speaker headings and assign by position.

Each canonical person gets a stable ``SPEAKER_XX`` (indexed by first appearance in the
protocol). Both files are rewritten with those tags (transcript ``<SD-SPK>SPEAKER_XX</SD>``,
protocol ``## SPEAKER_XX:``) and a per-session mapping report is written for inspection.
Pairs transcript↔protocol by normalised filename stem. Exit codes 0/1/2.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz
from tqdm import tqdm

# Reuse the main pipeline's helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_dataset import normalise_stem  # noqa: E402
from preprocess_protocol import split_front_matter  # noqa: E402

SPK_TAG_RE = re.compile(r"<SD-SPK>\s*(.*?)\s*</SD>")
SEG_LINE_RE = re.compile(r"^\[[0-9:.]+ --> [0-9:.]+\]\s*(.*)$")
# A protocol speaker heading: "## <name>:" optionally with a trailing "*" and —
# when docling glues the first sentence onto the heading line — trailing body text.
# group(1) = name, group(2) = any inline body that followed the colon.
PROT_HEADING_RE = re.compile(r"^##\s+(.+?)\s*:\s*\*?\s*(.*)$")

# Tokens stripped from a heading's name part to get the bare person name. Roles in
# parentheses are removed separately; these are the leading titles/offices.
TITLE_TOKENS = {
    "dr.", "prof.", "habil.", "dr", "prof",
    "präsidentin", "präsident", "vizepräsidentin", "vizepräsident",
    "alterspräsident", "alterspräsidentin",
}
# Give-the-floor trigger words: a chair turn must contain one of these before we trust
# a surname in it as "the next speaker".
CUE_TRIGGER_RE = re.compile(
    r"(das wort|hat das wort|spricht für|spricht jetzt|erteile|rufe|"
    r"bitten|bitte schön|bitte sehr|am mikrofon|das wort geht an|"
    r"fortsetzen|fort\.|setzen die aussprache)",
    re.IGNORECASE,
)
# Headings that are never speakers even if they end with a colon.
NON_SPEAKER_PREFIX_RE = re.compile(r"^(TOP\b|Beginn der Sitzung|Zu TOP|Inhalt$)", re.IGNORECASE)
# Lowercase particles allowed inside a surname ("André von Ossowski").
NAME_PARTICLES = {"von", "van", "de", "der", "den", "zu", "zur", "zum", "la", "le", "di", "da"}
NAME_TOKEN_RE = re.compile(r"^[A-ZÄÖÜ][\wäöüß.\-']*$")


# --------------------------------------------------------------------------- parsing

def parse_transcript(text: str) -> list[dict]:
    """Return turns ``[{label, text, lines}]`` in order from a diarised transcript."""
    _, body = split_front_matter(text)
    turns: list[dict] = []
    cur: dict | None = None
    for line in body.splitlines():
        m = SPK_TAG_RE.match(line.strip())
        if m:
            cur = {"label": m.group(1), "text_parts": [], "lines": []}
            turns.append(cur)
            continue
        if cur is None:
            continue
        cur["lines"].append(line)
        seg = SEG_LINE_RE.match(line.strip())
        if seg and seg.group(1).strip():
            cur["text_parts"].append(seg.group(1).strip())
    for t in turns:
        t["text"] = " ".join(t["text_parts"])
    return turns


def _split_name_role(heading_name: str) -> tuple[str, str]:
    """Split "André von Ossowski (fraktionslos)" -> ("André von Ossowski", "fraktionslos")."""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", heading_name)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return heading_name.strip(), ""


def canonical_name(heading_name: str) -> tuple[str, str]:
    """Return (canonical_key, surname) for a speaker heading name.

    Strips the parenthetical role/party and leading titles so role-variants of the same
    person collapse (e.g. "Vizepräsident Rainer Genilke" and "Rainer Genilke (CDU)" ->
    key "rainer genilke", surname "genilke").
    """
    bare, _role = _split_name_role(heading_name)
    tokens = [t for t in bare.split() if t.lower() not in TITLE_TOKENS]
    if not tokens:
        tokens = bare.split()
    key = " ".join(tokens).lower()
    surname = tokens[-1].lower() if tokens else key
    return key, surname


def looks_like_person_name(heading_name: str) -> bool:
    """True if the (paren-stripped, title-stripped) heading reads like a person name.

    Rejects section headers that merely end with a colon (e.g. "Fünf Argumente dazu",
    "Namens der Landesregierung beantwortet der Minister …"): a real name is 1–5 tokens,
    each capitalised (or an allowed lowercase particle), ending in a capitalised surname.
    """
    bare, _ = _split_name_role(heading_name)
    tokens = [t for t in bare.split() if t.lower() not in TITLE_TOKENS]
    if not (1 <= len(tokens) <= 5):
        return False
    if not NAME_TOKEN_RE.match(tokens[-1]):  # surname must be capitalised
        return False
    for t in tokens:
        if NAME_TOKEN_RE.match(t) or t.lower() in NAME_PARTICLES:
            continue
        return False
    return True


def is_speaker_heading(name: str) -> bool:
    if name.startswith("(") or NON_SPEAKER_PREFIX_RE.match(name):
        return False
    return looks_like_person_name(name)


def parse_protocol(text: str) -> tuple[list[dict], list[str], dict[str, str]]:
    """Parse a protocol into speaker blocks.

    Returns ``(blocks, order, key_to_display)`` where blocks is ``[{key, surname,
    display, text}]`` in document order, order is the list of canonical keys in
    first-appearance order, and key_to_display maps key -> a representative heading name.
    """
    _, body = split_front_matter(text)
    blocks: list[dict] = []
    order: list[str] = []
    key_to_display: dict[str, str] = {}
    cur: dict | None = None
    for line in body.splitlines():
        hm = PROT_HEADING_RE.match(line)
        name = hm.group(1).strip() if hm else None
        if name and is_speaker_heading(name):
            key, surname = canonical_name(name)
            cur = {"key": key, "surname": surname, "display": name, "text_parts": []}
            blocks.append(cur)
            inline = hm.group(2).strip()
            if inline:  # docling glued the first sentence onto the heading line
                cur["text_parts"].append(inline)
            if key not in key_to_display:
                key_to_display[key] = name
                order.append(key)
            continue
        if cur is not None:
            cur["text_parts"].append(line)
    for b in blocks:
        b["text"] = " ".join(p for p in b["text_parts"] if p.strip())
    return blocks, order, key_to_display


# --------------------------------------------------------------- matching (cascade)

def chair_naming_votes(turns: list[dict], surname_to_key: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Method 1: votes ``label -> {canonical_key: count}`` from chair give-the-floor cues.

    If turn i contains a trigger and a known surname, the *next* turn's label is voted to
    the canonical key of the last known surname appearing in turn i.
    """
    votes: dict[str, dict[str, int]] = {}
    for i, turn in enumerate(turns[:-1]):
        text = turn["text"]
        if not CUE_TRIGGER_RE.search(text):
            continue
        # last known surname mentioned in this (chair) turn
        chosen_key = None
        lowered = text.lower()
        best_pos = -1
        for surname, keys in surname_to_key.items():
            for m in re.finditer(rf"\b{re.escape(surname)}\b", lowered):
                if m.start() > best_pos:
                    best_pos, chosen_key = m.start(), keys[0]
        if chosen_key is None:
            continue
        nxt = turns[i + 1]["label"]
        votes.setdefault(nxt, {}).setdefault(chosen_key, 0)
        votes[nxt][chosen_key] += 1
    return votes


def content_match(label_text: str, key_to_text: dict[str, str], *, limit: int = 4000) -> tuple[str | None, float]:
    """Method 2: best canonical key by fuzzy text similarity, with its score."""
    if not label_text.strip():
        return None, 0.0
    lt = label_text[:limit]
    best_key, best_score = None, 0.0
    for key, text in key_to_text.items():
        if not text.strip():
            continue
        score = fuzz.token_set_ratio(lt, text[:limit])
        if score > best_score:
            best_score, best_key = score, key
    return best_key, best_score


def resolve_labels(
    turns: list[dict],
    blocks: list[dict],
    order: list[str],
    *,
    content_threshold: float,
) -> dict[str, dict]:
    """Run the priority cascade; return ``label -> {key, method, score}`` per pyannote label."""
    labels = list(dict.fromkeys(t["label"] for t in turns))  # first-appearance order

    surname_to_key: dict[str, list[str]] = {}
    for b in blocks:
        surname_to_key.setdefault(b["surname"], [])
        if b["key"] not in surname_to_key[b["surname"]]:
            surname_to_key[b["surname"]].append(b["key"])

    key_to_text: dict[str, str] = {}
    for b in blocks:
        key_to_text.setdefault(b["key"], "")
        key_to_text[b["key"]] += " " + b["text"]

    label_text: dict[str, str] = {}
    for t in turns:
        label_text.setdefault(t["label"], "")
        label_text[t["label"]] += " " + t["text"]

    resolved: dict[str, dict] = {}

    # 1. chair naming
    votes = chair_naming_votes(turns, surname_to_key)
    for label, vote in votes.items():
        key = max(vote, key=vote.get)
        resolved[label] = {"key": key, "method": "chair", "score": float(vote[key])}

    # 2. content matching
    for label in labels:
        if label in resolved:
            continue
        key, score = content_match(label_text.get(label, ""), key_to_text)
        if key is not None and score >= content_threshold:
            resolved[label] = {"key": key, "method": "content", "score": round(score, 1)}

    # 3. sequence / Rednerliste fallback
    # transcript blocks (collapse consecutive identical labels) vs protocol key order
    seq_labels: list[str] = []
    for t in turns:
        if not seq_labels or seq_labels[-1] != t["label"]:
            seq_labels.append(t["label"])
    for label in labels:
        if label in resolved:
            continue
        idx = seq_labels.index(label) if label in seq_labels else len(order)
        key = order[min(idx, len(order) - 1)] if order else None
        resolved[label] = {"key": key, "method": "sequence", "score": 0.0}

    return resolved


# ------------------------------------------------------------------- numbering / emit

def assign_speaker_ids(order: list[str], resolved: dict[str, dict]) -> dict[str, str]:
    """Map canonical key -> SPEAKER_XX (protocol first-appearance order, then extras)."""
    ids: dict[str, str] = {}
    n = 0
    for key in order:
        ids[key] = f"SPEAKER_{n:02d}"
        n += 1
    # canonical keys that only came from the transcript (or None) get trailing ids
    for info in resolved.values():
        key = info["key"]
        if key is None:
            continue
        if key not in ids:
            ids[key] = f"SPEAKER_{n:02d}"
            n += 1
    return ids


def rewrite_transcript(text: str, label_to_id: dict[str, str]) -> str:
    """Relabel <SD-SPK> tags and merge consecutive same-id turns."""
    front, body = split_front_matter(text)
    turns = parse_transcript(text)
    out: list[str] = []
    last_id: str | None = None
    for t in turns:
        sid = label_to_id.get(t["label"], "SPEAKER_UNK")
        if sid != last_id:
            if out:
                out.append("")
            out.append(f"<SD-SPK>{sid}</SD>")
            last_id = sid
        out.extend(t["lines"])
    return (front + "\n".join(out).strip() + "\n") if front else ("\n".join(out).strip() + "\n")


def rewrite_protocol(text: str, key_to_id: dict[str, str]) -> str:
    """Replace each speaker heading "## Name:" with "## SPEAKER_XX:"."""
    front, body = split_front_matter(text)
    out_lines: list[str] = []
    for line in body.splitlines():
        hm = PROT_HEADING_RE.match(line)
        name = hm.group(1).strip() if hm else None
        if name and is_speaker_heading(name):
            key, _ = canonical_name(name)
            sid = key_to_id.get(key, "SPEAKER_UNK")
            out_lines.append(f"## {sid}:")
            inline = hm.group(2).strip()
            if inline:  # preserve a sentence docling glued onto the heading line
                out_lines.append(inline)
        else:
            out_lines.append(line)
    return front + "\n".join(out_lines).rstrip() + "\n"


def build_report(resolved, key_to_id, key_to_display, label_to_id) -> dict:
    rows = []
    for label, info in sorted(label_to_id.items()):
        key = resolved.get(label, {}).get("key")
        rows.append({
            "pyannote_label": label,
            "speaker_id": label_to_id[label],
            "canonical_key": key,
            "protocol_name": key_to_display.get(key) if key else None,
            "method": resolved.get(label, {}).get("method"),
            "score": resolved.get(label, {}).get("score"),
        })
    protocol_speakers = [
        {"speaker_id": key_to_id[k], "name": key_to_display.get(k, k)}
        for k in key_to_id
    ]
    return {"transcript_labels": rows, "protocol_speakers": protocol_speakers}


# ------------------------------------------------------------------------------ main

def pair_files(transcript_dir: Path, protocol_dir: Path) -> list[tuple[Path, Path]]:
    prots = {normalise_stem(p.stem): p for p in protocol_dir.glob("*.md")}
    pairs = []
    for t in sorted(transcript_dir.glob("*.md")):
        key = normalise_stem(t.stem)
        if key in prots:
            pairs.append((t, prots[key]))
        else:
            print(f"no protocol match for {t.name}", file=sys.stderr)
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcript-dir", type=Path, default=Path("data_example/transcripts/md"))
    p.add_argument("--protocol-dir", type=Path, default=Path("data_example/protocols/md_clean"))
    p.add_argument("--out-transcript-dir", type=Path, default=Path("data_example/transcripts/md_anon"))
    p.add_argument("--out-protocol-dir", type=Path, default=Path("data_example/protocols/md_anon"))
    p.add_argument("--report-dir", type=Path, default=Path("data_example/speaker_maps"))
    p.add_argument("--content-threshold", type=float, default=60.0,
                   help="Min rapidfuzz token_set_ratio for a content match (default: 60)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    pairs = pair_files(args.transcript_dir, args.protocol_dir)
    if not pairs:
        print("no transcript/protocol pairs found", file=sys.stderr)
        return 1

    for d in (args.out_transcript_dir, args.out_protocol_dir, args.report_dir):
        d.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for tpath, ppath in tqdm(pairs, desc="match-speakers", unit="pair"):
        out_t = args.out_transcript_dir / tpath.name
        out_p = args.out_protocol_dir / ppath.name
        if out_t.exists() and out_p.exists() and not args.overwrite:
            continue
        try:
            ttext = tpath.read_text(encoding="utf-8")
            ptext = ppath.read_text(encoding="utf-8")
            turns = parse_transcript(ttext)
            blocks, order, key_to_display = parse_protocol(ptext)
            if not turns or not blocks:
                raise ValueError(f"empty parse (turns={len(turns)}, blocks={len(blocks)})")
            resolved = resolve_labels(turns, blocks, order,
                                      content_threshold=args.content_threshold)
            key_to_id = assign_speaker_ids(order, resolved)
            label_to_id = {lab: key_to_id.get(info["key"], "SPEAKER_UNK")
                           for lab, info in resolved.items()}
            out_t.write_text(rewrite_transcript(ttext, label_to_id), encoding="utf-8")
            out_p.write_text(rewrite_protocol(ptext, key_to_id), encoding="utf-8")
            report = build_report(resolved, key_to_id, key_to_display, label_to_id)
            (args.report_dir / f"{normalise_stem(tpath.stem)}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            methods = {}
            for r in report["transcript_labels"]:
                methods[r["method"]] = methods.get(r["method"], 0) + 1
            tqdm.write(f"{tpath.name}: {len(label_to_id)} labels -> "
                       f"{len(set(label_to_id.values()))} speakers ({methods})")
        except Exception as exc:
            failures.append((tpath.name, repr(exc)))
            print(f"\nERROR on {tpath.name}: {exc!r}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} pair(s) failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
