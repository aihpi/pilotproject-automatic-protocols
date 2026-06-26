#!/usr/bin/env python3
"""Resolve diarisation labels to real speaker names in committee transcripts.

The new transcripts carry generic ``<SD-SPK>SPEAKER_NN</SD>`` labels; the paired
protocol attributes speech as indirect-speech prose ("Dr. Benjamin Grimm
(Minister …) betont …"). To teach the model that attribution, each ``SPEAKER_NN``
must be mapped to the real name+role, which is then substituted into the transcript.

Resolution cascade (a label fixed by a higher tier is never overridden):

1. **Self-introduction** — "Mein Name ist André von Ossowski" fixes that turn's label.
2. **Chair give-the-floor cues** — when the chair clearly hands off to a single,
   unambiguous surname ("… Herr Hanko, bitte."), the *next* turn's label is voted to
   that person. Turns with pronoun/queue deferral ("Sie", "Ihnen", "auf meiner Liste",
   "wir sammeln") are skipped — they are the ambiguous cases. A label that collects
   votes for >1 person (diarisation merged speakers) is flagged and left unresolved.
3. **Content match** (secondary) — fuzzy-match a label's utterances against the protocol
   sentences mentioning each candidate; accepted only above ``--content-threshold``.
4. **LLM** (opt-in ``--use-llm``) — per TOP, ascending, ask the model to map the
   still-unresolved labels using that item's transcript + protocol; accept only
   ``confidence == "certain"``.

Outputs: prepared transcripts (resolved labels replaced with "Full Name (role)",
unresolved kept as ``SPEAKER_NN``), per-session ``speaker_maps/{stem}.json`` reports,
and an exclusions manifest ``{stem: [tops]}`` listing TOPs that still contain an
unresolved speaker (to be dropped from training). Requires TOP-tagged transcripts
(run ``tag_transcript_tops.py`` first).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

from utils import llm_utils
from build_dataset import normalise_stem, split_protocol_by_top
from preprocess_protocol import clean_protocol, split_front_matter
from utils.speaker_utils import (
    CUE_TRIGGER_RE,
    DEFER_SIGNAL_RE,
    SEG_LINE_RE,
    SPK_TAG_RE,
    canonical_name,
    extract_speaker_directory,
    extract_vorsitz_keys,
    format_speaker,
)

TOP_TAG_RE = re.compile(r"<SD-TOP>.*?(\d+).*?</SD>")
SELF_INTRO_RE = re.compile(
    r"(?i)(?:mein name ist|ich heiße|ich heisse|ich bin)\s+"
    r"([A-ZÄÖÜ][\wäöüß.\-']*(?:\s+(?:[A-ZÄÖÜ][\wäöüß.\-']*|von|van|de|der|zu))*)")
MAX_UNRESOLVED_SENTENCES = 12  # exclude a TOP if an unresolved speaker exceeds this many sentences
_SENTENCE_RE = re.compile(r"[.!?]+")


def count_sentences(text: str) -> int:
    return len(_SENTENCE_RE.findall(text))


# --------------------------------------------------------------------------- parsing

def parse_turns_with_top(text: str) -> list[dict]:
    """Parse turns ``[{label, text, top}]``, tracking the active ``<SD-TOP>``."""
    _, body = split_front_matter(text)
    turns: list[dict] = []
    cur: dict | None = None
    top: int | None = None
    for line in body.splitlines():
        s = line.strip()
        tm = TOP_TAG_RE.match(s)
        if tm:
            top = int(tm.group(1))
            continue
        m = SPK_TAG_RE.match(s)
        if m:
            cur = {"label": m.group(1), "text_parts": [], "top": top}
            turns.append(cur)
            continue
        if cur is None:
            continue
        seg = SEG_LINE_RE.match(s)
        if seg and seg.group(1).strip():
            cur["text_parts"].append(seg.group(1).strip())
    for t in turns:
        t["text"] = " ".join(t["text_parts"])
    return turns


def flatten_directory(directory: dict[str, list[dict]]) -> dict[str, dict]:
    return {p["key"]: p for people in directory.values() for p in people}


# ------------------------------------------------------------------ heuristic tiers

def _surname_hits(text: str, directory: dict[str, list[dict]]) -> list[tuple[int, dict]]:
    """Positions of unambiguous (single-person) known surnames in ``text``."""
    hits: list[tuple[int, dict]] = []
    lowered = text.lower()
    for surname, people in directory.items():
        if len(people) != 1 or len(surname) < 3:
            continue
        for m in re.finditer(rf"\b{re.escape(surname)}\b", lowered):
            hits.append((m.start(), people[0]))
    return hits


def resolve_self_intro(turns: list[dict], directory: dict[str, list[dict]],
                       resolved: dict[str, dict]) -> None:
    for t in turns:
        if t["label"] in resolved:
            continue
        m = SELF_INTRO_RE.search(t["text"])
        if not m:
            continue
        _, surname = canonical_name(m.group(1))
        people = directory.get(surname)
        if people and len(people) == 1:
            resolved[t["label"]] = {"key": people[0]["key"], "method": "self-intro",
                                    "score": 1.0}


def chair_cue_votes(turns: list[dict],
                    directory: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    """Vote ``label -> {key: count}`` from clean chair give-the-floor cues."""
    votes: dict[str, dict[str, int]] = {}
    for i in range(len(turns) - 1):
        text = turns[i]["text"]
        if not CUE_TRIGGER_RE.search(text) or DEFER_SIGNAL_RE.search(text):
            continue
        if turns[i]["label"] == turns[i + 1]["label"]:
            continue
        hits = _surname_hits(text, directory)
        if not hits:
            continue
        person = max(hits, key=lambda h: h[0])[1]  # surname nearest the end of the turn
        nxt = turns[i + 1]["label"]
        votes.setdefault(nxt, {})
        votes[nxt][person["key"]] = votes[nxt].get(person["key"], 0) + 1
    return votes


def apply_chair_votes(votes: dict[str, dict[str, int]], resolved: dict[str, dict],
                      conflicts: dict[str, dict]) -> None:
    for label, vote in votes.items():
        if label in resolved:
            continue
        ranked = sorted(vote.items(), key=lambda kv: kv[1], reverse=True)
        top_key, top_n = ranked[0]
        if len(ranked) > 1:
            conflicts[label] = dict(vote)
            if top_n <= ranked[1][1]:  # no clear winner -> leave for the LLM
                continue
        resolved[label] = {"key": top_key, "method": "chair", "score": float(top_n),
                           "conflict": len(ranked) > 1}


def resolve_chair(turns: list[dict], vorsitz_keys: list[str],
                  resolved: dict[str, dict]) -> None:
    """Map the busiest still-unresolved cue-issuing label(s) to the cover's Vorsitz name(s).

    The chair is never announced by name but is named under "Vorsitz:" on the cover and
    issues most give-the-floor cues. Assign by cue frequency to the unassigned Vorsitz
    persons (order: most-cues label -> first unassigned Vorsitz key)."""
    cue_counts: dict[str, int] = {}
    for i in range(len(turns) - 1):
        if CUE_TRIGGER_RE.search(turns[i]["text"]):
            cue_counts[turns[i]["label"]] = cue_counts.get(turns[i]["label"], 0) + 1
    assigned = {info["key"] for info in resolved.values()}
    pending = [k for k in vorsitz_keys if k not in assigned]
    busiest = [lab for lab, _ in sorted(cue_counts.items(), key=lambda kv: kv[1], reverse=True)
               if lab not in resolved and cue_counts[lab] >= 2]
    for lab, key in zip(busiest, pending):
        resolved[lab] = {"key": key, "method": "chair-vorsitz", "score": float(cue_counts[lab])}


def content_match(turns: list[dict], body: str, directory: dict[str, list[dict]],
                  resolved: dict[str, dict], threshold: float) -> None:
    label_text: dict[str, str] = {}
    for t in turns:
        label_text[t["label"]] = (label_text.get(t["label"], "") + " " + t["text"])[:6000]
    sentences = re.split(r"(?<=[.!?])\s+", body)
    key_to_text: dict[str, str] = {}
    for surname, people in directory.items():
        if len(people) != 1:
            continue
        ref = " ".join(s for s in sentences
                       if re.search(rf"\b{re.escape(surname)}\b", s, re.I))
        if ref:
            key_to_text[people[0]["key"]] = ref[:6000]
    assigned = {info["key"] for info in resolved.values()}
    for label, ltext in label_text.items():
        if label in resolved or not ltext.strip():
            continue
        best_key, best = None, 0.0
        for key, ref in key_to_text.items():
            if key in assigned:
                continue
            score = fuzz.token_set_ratio(ltext, ref)
            if score > best:
                best, best_key = score, key
        if best_key is not None and best >= threshold:
            resolved[label] = {"key": best_key, "method": "content", "score": round(best, 1)}
            assigned.add(best_key)


# ------------------------------------------------------------------------- LLM tier

def llm_resolve(client, model: str, turns: list[dict], pr_tops: dict[int, str],
                directory: dict[str, list[dict]], resolved: dict[str, dict],
                max_tokens: int) -> None:
    """Per-TOP escalation: map still-unresolved labels via the LLM, accept only certain."""
    key_to_person = flatten_directory(directory)
    tx_tops = sorted({t["top"] for t in turns if t["top"] is not None})
    for n in tx_tops:
        unresolved = sorted({t["label"] for t in turns
                             if t["top"] == n and t["label"] not in resolved and t["text"]})
        if not unresolved:
            continue
        utt: dict[str, list[str]] = {}
        for t in turns:
            if t["top"] == n and t["label"] in unresolved:
                utt.setdefault(t["label"], []).append(t["text"])
        assigned = {info["key"] for info in resolved.values()}
        candidates = [format_speaker(p) for k, p in key_to_person.items() if k not in assigned]
        known = {lab: format_speaker(key_to_person[i["key"]]) for lab, i in resolved.items()
                 if i["key"] in key_to_person}
        prompt = _build_llm_prompt(n, utt, pr_tops.get(n, ""), candidates, known)
        mapping = llm_utils.chat_json(client, model, prompt, max_tokens=max_tokens)
        for label, info in mapping.items():
            if label in resolved or not isinstance(info, dict):
                continue
            if str(info.get("confidence", "")).lower() != "certain":
                continue
            key = _match_candidate(info.get("name", ""), key_to_person, assigned)
            if key:
                resolved[label] = {"key": key, "method": "llm", "score": 1.0}
                assigned.add(key)
        if all(t["label"] in resolved for t in turns if t["text"]):
            break


def _build_llm_prompt(top: int, utt: dict[str, list[str]], protocol: str,
                      candidates: list[str], known: dict[str, str]) -> str:
    lines = [f"Du ordnest Sprecher-Labels einer Ausschusssitzung zu (Tagesordnungspunkt {top}).",
             "Nutze das Protokoll (indirekte Rede mit Namen) und die Transkript-Redebeiträge.",
             "", "PROTOKOLL:", protocol[:8000], ""]
    if known:
        lines += ["BEREITS ZUGEORDNET (nicht erneut vergeben):",
                  "; ".join(f"{lab}={name}" for lab, name in known.items()), ""]
    lines += ["NOCH OFFENE TRANSKRIPT-REDEBEITRÄGE:"]
    for label, parts in utt.items():
        lines.append(f"[{label}] " + " ".join(parts)[:2500])
    lines += ["", "MÖGLICHE NAMEN:", "; ".join(candidates) or "(keine)", "",
              "Gib NUR JSON zurück: {\"SPEAKER_NN\": {\"name\": \"<voller Name aus der Liste "
              "oder null>\", \"confidence\": \"certain|unsure\"}}. Verwende \"certain\" nur, "
              "wenn die Zuordnung eindeutig aus den Inhalten folgt."]
    return "\n".join(lines)


def _match_candidate(name: str, key_to_person: dict[str, dict], assigned: set) -> str | None:
    if not name or name.lower() in ("null", "none", ""):
        return None
    key, _ = canonical_name(name)
    if key in key_to_person and key not in assigned:
        return key
    best_key, best = None, 0.0
    for k, p in key_to_person.items():
        if k in assigned:
            continue
        score = fuzz.token_set_ratio(name.lower(), p["full"].lower())
        if score > best:
            best, best_key = score, k
    return best_key if best >= 85 else None


# --------------------------------------------------------------- transcript-LLM tier

def llm_resolve_from_transcript(client, model: str, turns: list[dict],
                                resolved: dict[str, dict], key_to_person: dict[str, dict],
                                max_tokens: int) -> int:
    """Identify still-unresolved speakers from the TRANSCRIPT itself (not the protocol
    directory).

    The directory tiers can only assign names that appear in the protocol; guests/experts
    who speak but are absent from the protocol stay ``SPEAKER_NN``. Here the model reads the
    transcript context — the chair's give-the-floor announcement (usually at the END of the
    previous turn, e.g. "Herr Klemm, bitte"), self-introductions, or direct address — and
    names the speaker verbatim from the transcript. Synthesises a person entry (so it flows
    through the normal rewrite/report) and accepts only ``confidence == "certain"``. Returns
    the number of newly resolved labels."""
    unresolved = sorted({t["label"] for t in turns if t["label"] not in resolved and t["text"]})
    if not unresolved:
        return 0
    utt: dict[str, list[str]] = {}
    handoff: dict[str, list[str]] = {}
    for i, t in enumerate(turns):
        if t["label"] not in unresolved or not t["text"]:
            continue
        if len(" ".join(utt.get(t["label"], []))) < 1500:
            utt.setdefault(t["label"], []).append(t["text"])
        if i > 0 and turns[i - 1]["label"] != t["label"]:
            tail = turns[i - 1]["text"][-300:].strip()
            if tail and tail not in handoff.get(t["label"], []):
                handoff.setdefault(t["label"], []).append(tail)
    try:
        mapping = llm_utils.chat_json(client, model, _build_transcript_llm_prompt(utt, handoff),
                                      max_tokens=max_tokens)
    except Exception:
        return 0
    assigned = {info["key"] for info in resolved.values()}
    n_new = 0
    for label, info in mapping.items():
        if label in resolved or not isinstance(info, dict):
            continue
        if str(info.get("confidence", "")).lower() != "certain":
            continue
        name = (info.get("name") or "").strip()
        if not name or name.lower() in ("null", "none", "unbekannt"):
            continue
        role = (info.get("role") or "").strip()
        if role.lower() in ("null", "none"):
            role = ""
        # If the transcript name matches a directory person earlier tiers missed (e.g. an
        # ASR spelling variant of a cover member), snap to that canonical entry; otherwise
        # synthesise a new person (a genuine guest absent from the protocol).
        dkey = _match_candidate(name, key_to_person, assigned)
        if dkey:
            key = dkey
        else:
            key, surname = canonical_name(name)
            if not key or key in assigned:       # skip empties + names already taken
                continue
            key_to_person.setdefault(key, {"full": name, "role": role,
                                           "key": key, "surname": surname})
        resolved[label] = {"key": key, "method": "transcript-llm", "score": 1.0}
        assigned.add(key)
        n_new += 1
    return n_new


def _build_transcript_llm_prompt(utt: dict[str, list[str]],
                                 handoff: dict[str, list[str]]) -> str:
    lines = [
        "Du identifizierst Sprecher in einem Ausschuss-Transkript ANHAND DES TRANSKRIPTS.",
        "Diese Sprecher stehen NICHT in einer Namensliste; ermittle den Namen aus dem Kontext:",
        "- Die/der Vorsitzende kündigt die nächste Person oft am ENDE des vorherigen Beitrags "
        "an („Herr Klemm, bitte.“, „jetzt hat Frau Sademach das Wort“).",
        "- Selbstvorstellung („mein Name ist …“) oder direkte Anrede durch andere.",
        "Gib den Namen WÖRTLICH wie im Transkript an (inkl. Anrede/Titel, falls genannt) und die "
        "Rolle/Funktion, falls genannt.",
        "",
    ]
    for label in utt:
        lines.append(f"[{label}]")
        if handoff.get(label):
            lines.append("  ANKÜNDIGUNG (Ende des vorherigen Beitrags): "
                         + (" | ".join(handoff[label]))[:600])
        lines.append("  REDEBEITRAG: " + (" ".join(utt[label]))[:1500])
    lines += ["",
              'Gib NUR JSON zurück: {"SPEAKER_NN": {"name": "<Name aus dem Transkript oder null>", '
              '"role": "<Rolle/Funktion oder null>", "confidence": "certain|unsure"}}. '
              'Verwende "certain" nur, wenn der Name eindeutig aus dem Kontext hervorgeht. Rate nicht.']
    return "\n".join(lines)


# ------------------------------------------------------------------- emit / exclude

def rewrite_transcript(text: str, label_to_name: dict[str, str]) -> str:
    """Replace resolved ``<SD-SPK>`` labels with names; keep <SD-TOP>/segments/unresolved."""
    front, body = split_front_matter(text)
    out: list[str] = []
    for line in body.splitlines():
        m = SPK_TAG_RE.match(line.strip())
        if m and m.group(1) in label_to_name:
            out.append(f"<SD-SPK>{label_to_name[m.group(1)]}</SD>")
        else:
            out.append(line)
    return front + "\n".join(out).strip() + "\n"


def excluded_tops(turns: list[dict], resolved: dict[str, dict], max_sentences: int) -> list[int]:
    """TOPs with an unresolved speaker whose turn has more than ``max_sentences`` sentences."""
    bad: set[int] = set()
    for t in turns:
        if (t["top"] is not None and t["label"] not in resolved
                and count_sentences(t["text"]) > max_sentences):
            bad.add(t["top"])
    return sorted(bad)


def build_report(turns: list[dict], resolved: dict[str, dict], conflicts: dict[str, dict],
                 key_to_person: dict[str, dict], excluded: list[int]) -> dict:
    labels = sorted({t["label"] for t in turns})
    rows = []
    for label in labels:
        info = resolved.get(label)
        person = key_to_person.get(info["key"]) if info else None
        rows.append({
            "label": label,
            "name": format_speaker(person) if person else None,
            "method": info["method"] if info else None,
            "score": info.get("score") if info else None,
            "conflict": conflicts.get(label),
        })
    return {"labels": rows, "excluded_tops": excluded,
            "resolved": sum(1 for label in labels if label in resolved),
            "total_labels": len(labels)}


# ------------------------------------------------------------------------------ main

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


def process_pair(tpath: Path, ppath: Path, *, client, model: str,
                 content_threshold: float, max_sentences: int, max_tokens: int,
                 transcript_llm: bool = False) -> dict:
    ttext = tpath.read_text(encoding="utf-8")
    cover, body, _ = clean_protocol(ppath.read_text(encoding="utf-8"))
    directory = extract_speaker_directory(cover, body)
    key_to_person = flatten_directory(directory)
    turns = parse_turns_with_top(ttext)
    if not turns or not directory:
        raise ValueError(f"empty parse (turns={len(turns)}, names={len(key_to_person)})")

    resolved: dict[str, dict] = {}
    conflicts: dict[str, dict] = {}
    resolve_self_intro(turns, directory, resolved)
    apply_chair_votes(chair_cue_votes(turns, directory), resolved, conflicts)
    resolve_chair(turns, extract_vorsitz_keys(cover, directory), resolved)
    content_match(turns, body, directory, resolved, content_threshold)
    if client is not None:
        llm_resolve(client, model, turns, split_protocol_by_top(body), directory, resolved,
                    max_tokens)
        if transcript_llm:  # recover guests absent from the protocol, named only in the transcript
            llm_resolve_from_transcript(client, model, turns, resolved, key_to_person, max_tokens)

    label_to_name = {lab: format_speaker(key_to_person[info["key"]])
                     for lab, info in resolved.items() if info["key"] in key_to_person}
    excl = excluded_tops(turns, resolved, max_sentences)
    return {
        "stem": normalise_stem(tpath.stem),
        "name": tpath.name,
        "report_name": tpath.stem,
        "prepared": rewrite_transcript(ttext, label_to_name),
        "report": build_report(turns, resolved, conflicts, key_to_person, excl),
        "excluded": excl,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcript-dir", type=Path,
                   default=Path("data_example02/transcripts/md_top"))
    p.add_argument("--protocol-dir", type=Path, default=Path("data_example02/protocols/md"))
    p.add_argument("--out-transcript-dir", type=Path,
                   default=Path("data_example02/transcripts/md_prepared"))
    p.add_argument("--report-dir", type=Path, default=Path("data_example02/speaker_maps"))
    p.add_argument("--exclusions-out", type=Path, default=Path("data_example02/exclusions.json"))
    p.add_argument("--content-threshold", type=float, default=75.0,
                   help="Min rapidfuzz score for a (secondary) content match (default: 75)")
    p.add_argument("--max-unresolved-sentences", type=int, default=MAX_UNRESOLVED_SENTENCES,
                   help="A TOP is excluded only if an unresolved speaker has a turn with MORE "
                        f"than this many sentences (default: {MAX_UNRESOLVED_SENTENCES}); short "
                        "interjections by an unidentified speaker are tolerated")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip the LLM escalation (heuristics only)")
    p.add_argument("--llm-model", default=llm_utils.DEFAULT_MODEL)
    p.add_argument("--llm-base-url", default=None,
                   help="Override OPENAI_API_BASE (default: from environment)")
    p.add_argument("--max-tokens", type=int, default=llm_utils.DEFAULT_MAX_TOKENS,
                   help=f"Completion budget incl. reasoning (default: {llm_utils.DEFAULT_MAX_TOKENS})")
    p.add_argument("--concurrency", type=int, default=4, help="Parallel sessions (default: 4)")
    p.add_argument("--transcript-llm", action="store_true",
                   help="After the protocol-directory tiers, ask the LLM to name still-"
                        "unresolved speakers from the TRANSCRIPT itself (chair hand-offs, self-"
                        "introductions, direct address) — recovers guests absent from the protocol")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    pairs = pair_files(args.transcript_dir, args.protocol_dir)
    if not pairs:
        print("no transcript/protocol pairs found", file=sys.stderr)
        return 1

    use_llm = not args.no_llm and llm_utils.have_key()
    if not args.no_llm and not use_llm:
        print("OPENAI_API_KEY not set; resolving with heuristics only", file=sys.stderr)
    client = llm_utils.make_client(args.llm_base_url) if use_llm else None
    args.out_transcript_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    print(f"matching {len(pairs)} session(s) "
          f"({'LLM ' + args.llm_model if use_llm else 'heuristics only'}, "
          f"concurrency {args.concurrency})", file=sys.stderr)
    results = llm_utils.run_parallel(
        pairs,
        lambda pr: process_pair(pr[0], pr[1], client=client, model=args.llm_model,
                                content_threshold=args.content_threshold,
                                max_sentences=args.max_unresolved_sentences,
                                max_tokens=args.max_tokens,
                                transcript_llm=args.transcript_llm),
        args.concurrency)

    exclusions: dict[str, list[int]] = {}
    failures = 0
    for (tpath, _), res in zip(pairs, results):
        if isinstance(res, Exception):
            failures += 1
            print(f"ERROR on {tpath.name}: {res!r}", file=sys.stderr)
            continue
        (args.out_transcript_dir / res["name"]).write_text(res["prepared"], encoding="utf-8")
        (args.report_dir / f"{res['report_name']}.json").write_text(
            json.dumps(res["report"], ensure_ascii=False, indent=2), encoding="utf-8")
        if res["excluded"]:
            exclusions[res["stem"]] = res["excluded"]
        r = res["report"]
        print(f"{res['name']}: {r['resolved']}/{r['total_labels']} labels resolved, "
              f"excluded TOPs {res['excluded'] or '[]'}")

    args.exclusions_out.parent.mkdir(parents=True, exist_ok=True)
    args.exclusions_out.write_text(json.dumps(exclusions, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"\nwrote exclusions for {len(exclusions)} session(s) to {args.exclusions_out}",
          file=sys.stderr)
    if failures:
        print(f"{failures} pair(s) failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
