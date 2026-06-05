#!/usr/bin/env python3
"""Tag diarised transcripts with ``<SD-TOP>`` boundaries from the protocol agenda.

The committee transcripts carry no agenda markers — the chair announces each item
verbally, but inconsistently: with a number ("… Tagesordnungspunkt 3 …"), by title only
("… kommen wir zum nächsten Tagesordnungspunkt, die Konstituierung …"), or only when
*closing* the previous item. A pure regex over spoken numbers therefore misses items and
is poisoned when the chair previews higher numbers in the opening preamble.

By default this uses an **LLM** (gpt-oss-120b via the HPI endpoint) with global context:
it gets the full agenda, each item's protocol prose as a semantic anchor, and the whole
transcript as a compact numbered turn list, and assigns each agenda item to the turn where
its discussion begins. Long transcripts are processed in windows; sessions run in parallel.
``--no-llm`` (or a missing key) falls back to a transition-verb-gated regex.

``<SD-TOP>TOP N</SD>`` is inserted before the assigned turn's ``<SD-SPK>`` so the per-TOP
slice keeps that turn's speaker. Verification per session (printed + JSON report): cover vs
transcript TOP counts and which TOPs appear on only one side.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import llm_utils
from build_dataset import normalise_stem, split_protocol_by_top
from preprocess_protocol import clean_protocol, split_cover, split_front_matter
from speaker_utils import SEG_LINE_RE, SPK_TAG_RE

TOP_ANNOUNCE_RE = re.compile(r"(?i)\b(?:TOP|Tagesordnungspunkt)\s*(\d+)")
# Transition verbs that mark a genuine "we now turn to item N" (vs. a preamble preview
# of guests/items). Gating on these fixes the monotonic-poisoning failure.
TRANSITION_RE = re.compile(
    r"(?i)\b(rufe|aufrufen|aufzurufen|kommen|komme|beginnen|widmen|"
    r"weiterverfahren|fortfahren|nächste[rns]?|eintreten|eintritt)\b")
# Agenda anchor: a "## Tagesordnung" heading OR a "| Tagesordnung: |" table cell.
AGENDA_ANCHOR_RE = re.compile(r"(?im)Tagesordnung")
AGENDA_BULLET_RE = re.compile(r"^-\s*(\d+(?:\.\d+)?)\s+(.+)$")          # "- 1 Title"
AGENDA_TABLE_RE = re.compile(r"^\|\s*(\d+(?:\.\d+)?)\s*\|\s*([^|]+?)\s*\|")  # "| 1 | Title | …"
AGENDA_ORDINAL_RE = re.compile(r"^(\d+)\.\s+(\S.*)$")                   # "1. Title"
_NOT_AGENDA_TITLE_RE = re.compile(r"^(\(|\d|(?i:\(?(teilweise\s+)?(öffentlich|nicht)))")

TURN_TEXT_CAP = 300   # chars of each turn shown to the LLM
ANCHOR_CAP = 400      # chars of each protocol section used as an anchor


def _is_agenda_title(title: str) -> bool:
    return bool(title) and not _NOT_AGENDA_TITLE_RE.match(title)


def parse_agenda(cover_text: str) -> dict[int, str]:
    """Return ``{integer_top: title}`` from the cover agenda (heading or table)."""
    m = AGENDA_ANCHOR_RE.search(cover_text)
    if not m:
        return {}
    tops: dict[int, str] = {}
    for line in cover_text[m.end():].splitlines():
        s = line.strip()
        bm = (AGENDA_BULLET_RE.match(s) or AGENDA_TABLE_RE.match(s)
              or AGENDA_ORDINAL_RE.match(s))
        if not bm:
            continue
        title = bm.group(2).strip()
        if not _is_agenda_title(title):
            continue
        tops.setdefault(int(bm.group(1).split(".")[0]), title)
    return tops


# ------------------------------------------------------------------- turn parsing

def parse_turns(body: str) -> tuple[list[str], list[dict]]:
    """Split a transcript body into (preamble_lines, turns).

    Each turn is ``{label, lines, text}`` where ``lines`` are the verbatim body lines
    (starting with the ``<SD-SPK>`` line) and ``text`` is the joined segment text.
    """
    preamble: list[str] = []
    turns: list[dict] = []
    cur: dict | None = None
    for line in body.splitlines():
        m = SPK_TAG_RE.match(line.strip())
        if m:
            cur = {"label": m.group(1), "lines": [line], "text_parts": []}
            turns.append(cur)
            continue
        if cur is None:
            preamble.append(line)
            continue
        cur["lines"].append(line)
        seg = SEG_LINE_RE.match(line.strip())
        if seg and seg.group(1).strip():
            cur["text_parts"].append(seg.group(1).strip())
    for t in turns:
        t["text"] = " ".join(t["text_parts"])
    return preamble, turns


def rebuild(front: str, preamble: list[str], turns: list[dict],
            top_to_turn: dict[int, int]) -> tuple[str, list[int]]:
    """Insert ``<SD-TOP>`` before each assigned turn, enforcing monotonic order."""
    chosen = sorted((top, ti) for top, ti in top_to_turn.items() if ti is not None)
    valid: dict[int, int] = {}  # turn_index -> top
    last_ti = -1
    for top, ti in chosen:                 # strictly increasing turn index by TOP order
        if 0 <= ti < len(turns) and ti > last_ti:
            valid[ti] = top
            last_ti = ti
    out = list(preamble)
    for i, t in enumerate(turns):
        if i in valid:
            out.append(f"<SD-TOP>TOP {valid[i]}</SD>")
        out.extend(t["lines"])
    body = "\n".join(out).strip() + "\n"
    return (front + body) if front else body, sorted(valid.values())


# ----------------------------------------------------------------- regex detection

def detect_tops_regex(turns: list[dict]) -> dict[int, int]:
    """Transition-verb-gated, monotonic regex detection: ``{top: turn_index}``."""
    found: dict[int, int] = {}
    last = 0
    for i, t in enumerate(turns):
        if not TRANSITION_RE.search(t["text"]):
            continue
        for nm in TOP_ANNOUNCE_RE.finditer(t["text"]):
            n = int(nm.group(1))
            if n > last:
                found.setdefault(n, i)
                last = n
    return found


# ------------------------------------------------------------------- LLM detection

def _protocol_anchors(protocol_text: str) -> dict[int, str]:
    cover, body, _ = clean_protocol(protocol_text)
    return {n: re.sub(r"\s+", " ", txt)[:ANCHOR_CAP]
            for n, txt in split_protocol_by_top(body).items()}


def _top_prompt(agenda: dict[int, str], anchors: dict[int, str],
                pending: list[int], window: list[tuple[int, dict]]) -> str:
    lines = [
        "Du erhältst die Tagesordnung einer Ausschusssitzung und das Transkript als "
        "nummerierte Redebeiträge (Turns). Ordne JEDEM gelisteten Tagesordnungspunkt (TOP) "
        "den Turn-Index zu, an dem die BERATUNG dieses Punktes BEGINNT — dort, wo die "
        "Sitzungsleitung den Punkt aufruft oder inhaltlich einführt bzw. die zuständige "
        "Person um einen Bericht bittet. NICHT der Turn, in dem der Punkt geschlossen wird. "
        "Wird ein Punkt nicht ausdrücklich aufgerufen (z.B. wenn die Sitzung direkt mit dem "
        "Thema beginnt oder es nur narrativ eingeleitet wird), wähle den Turn, in dem das "
        "Thema (vgl. Protokoll-Anker) erstmals behandelt wird. Die Turn-Indizes müssen mit "
        "steigender TOP-Nummer nicht-fallend sein. Gib null NUR, wenn der Punkt in den "
        "gezeigten Turns gar nicht vorkommt (etwa der nichtöffentliche Teil).",
        "", "OFFENE TAGESORDNUNGSPUNKTE:"]
    for n in pending:
        anchor = anchors.get(n, "")
        lines.append(f"  TOP {n}: {agenda.get(n, '')}"
                     + (f"  | Protokoll-Anker: {anchor}" if anchor else ""))
    lines += ["", "TRANSKRIPT-TURNS:"]
    for idx, t in window:
        lines.append(f"T{idx} [{t['label']}] {t['text'][:TURN_TEXT_CAP]}")
    lines += ["", "Gib NUR JSON zurück; Schlüssel = TOP-Nummer als Zahl, Wert = Turn-Index "
              'als Zahl oder null. Beispiel: {"1": 11, "2": 20, "3": null}']
    return "\n".join(lines)


def detect_tops_llm(client, model: str, agenda: dict[int, str], anchors: dict[int, str],
                    turns: list[dict], token_budget: int, max_tokens: int) -> dict[int, int]:
    """Global single-pass assignment with a windowed fallback for long transcripts."""
    found: dict[int, int] = {}
    pending = sorted(agenda)
    base = llm_utils.estimate_tokens(_top_prompt(agenda, anchors, pending, []))
    start = 0
    while pending and start < len(turns):
        window: list[tuple[int, dict]] = []
        toks = base
        i = start
        while i < len(turns):
            t_tok = llm_utils.estimate_tokens(turns[i]["text"][:TURN_TEXT_CAP]) + 8
            if window and toks + t_tok > token_budget:
                break
            window.append((i, turns[i]))
            toks += t_tok
            i += 1
        res = llm_utils.chat_json(client, model,
                                  _top_prompt(agenda, anchors, pending, window),
                                  max_tokens=max_tokens)
        progressed = False
        for k, v in res.items():
            km = re.search(r"\d+", str(k))           # accept "1", "TOP 1", "TOP1"
            vm = re.search(r"\d+", str(v)) if v is not None else None  # "6", "T6", null
            if not km:
                continue
            top = int(km.group())
            ti = int(vm.group()) if vm else None
            if top in agenda and top not in found and ti is not None and start <= ti < i:
                found[top] = ti
                progressed = True
        pending = [n for n in sorted(agenda) if n not in found]
        if i >= len(turns):
            break  # whole transcript seen in this window
        start = (max(found.values()) + 1) if progressed else i
    return found


# ------------------------------------------------------------------------- per session

def process_pair(tpath: Path, ppath: Path, *, use_llm: bool, client, model: str,
                 token_budget: int, max_tokens: int) -> dict:
    ptext = ppath.read_text(encoding="utf-8")
    cover, _b, matched = split_cover(ptext)
    agenda = parse_agenda(cover if matched else ptext)
    front, body = split_front_matter(tpath.read_text(encoding="utf-8"))
    preamble, turns = parse_turns(body)

    if use_llm and agenda:
        top_to_turn = detect_tops_llm(client, model, agenda, _protocol_anchors(ptext),
                                      turns, token_budget, max_tokens)
    else:
        top_to_turn = detect_tops_regex(turns)
    tagged, found = rebuild(front, preamble, turns, top_to_turn)

    cover_tops, tx_tops = sorted(agenda), sorted(found)
    return {
        "stem": normalise_stem(tpath.stem),
        "name": tpath.name,
        "tagged": tagged,
        "cover_tops": cover_tops,
        "transcript_tops": tx_tops,
        "cover_only": sorted(set(cover_tops) - set(tx_tops)),
        "transcript_only": sorted(set(tx_tops) - set(cover_tops)),
        "agenda_titles": {str(k): v for k, v in sorted(agenda.items())},
        "method": "llm" if (use_llm and agenda) else "regex",
    }


def pair_files(transcript_dir: Path, protocol_dir: Path) -> list[tuple[Path, Path]]:
    prots = {normalise_stem(p.stem): p for p in protocol_dir.glob("*.md")}
    pairs: list[tuple[Path, Path]] = []
    for t in sorted(transcript_dir.glob("*.md")):
        key = normalise_stem(t.stem)
        if key in prots:
            pairs.append((t, prots[key]))
        else:
            print(f"no protocol match for {t.name}", file=sys.stderr)
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcript-dir", type=Path, default=Path("data_example02/transcripts/md"))
    p.add_argument("--protocol-dir", type=Path, default=Path("data_example02/protocols/md"))
    p.add_argument("--out-dir", type=Path, default=Path("data_example02/transcripts/md_top"))
    p.add_argument("--report-dir", type=Path, default=None,
                   help="Per-session TOP reports (default: <out-dir>/top_reports)")
    p.add_argument("--no-llm", action="store_true", help="Use the regex fallback, not the LLM")
    p.add_argument("--llm-model", default=llm_utils.DEFAULT_MODEL)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--concurrency", type=int, default=4, help="Parallel sessions (default: 4)")
    p.add_argument("--token-budget", type=int, default=24000,
                   help="Approx prompt-token budget before windowing (default: 24000)")
    p.add_argument("--max-tokens", type=int, default=llm_utils.DEFAULT_MAX_TOKENS,
                   help=f"Completion budget incl. reasoning (default: {llm_utils.DEFAULT_MAX_TOKENS})")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    pairs = pair_files(args.transcript_dir, args.protocol_dir)
    if not pairs:
        print("no transcript/protocol pairs found", file=sys.stderr)
        return 1
    if not args.overwrite:
        pairs = [(t, pr) for t, pr in pairs if not (args.out_dir / t.name).exists()]
        if not pairs:
            print("all outputs exist; use --overwrite", file=sys.stderr)
            return 0

    use_llm = not args.no_llm and llm_utils.have_key()
    if not args.no_llm and not use_llm:
        print("OPENAI_API_KEY not set; falling back to regex TOP detection", file=sys.stderr)
    client = llm_utils.make_client(args.llm_base_url) if use_llm else None

    report_dir = args.report_dir or (args.out_dir / "top_reports")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"tagging {len(pairs)} session(s) "
          f"({'LLM ' + args.llm_model if use_llm else 'regex'}, "
          f"concurrency {args.concurrency})", file=sys.stderr)
    results = llm_utils.run_parallel(
        pairs,
        lambda pr: process_pair(pr[0], pr[1], use_llm=use_llm, client=client,
                                model=args.llm_model, token_budget=args.token_budget,
                                max_tokens=args.max_tokens),
        args.concurrency)

    failures = 0
    for (tpath, _), res in zip(pairs, results):
        if isinstance(res, Exception):
            failures += 1
            print(f"ERROR on {tpath.name}: {res!r}", file=sys.stderr)
            continue
        (args.out_dir / res["name"]).write_text(res["tagged"], encoding="utf-8")
        (report_dir / f"{res['stem']}.json").write_text(
            json.dumps({k: res[k] for k in
                        ("stem", "cover_tops", "transcript_tops", "cover_only",
                         "transcript_only", "agenda_titles", "method")},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{res['name']}: cover {len(res['cover_tops'])} {res['cover_tops']}, "
              f"transcript {len(res['transcript_tops'])} {res['transcript_tops']}"
              + (f", cover-only {res['cover_only']}" if res['cover_only'] else "")
              + (f", transcript-only {res['transcript_only']}" if res['transcript_only'] else ""))

    if failures:
        print(f"\n{failures} session(s) failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
