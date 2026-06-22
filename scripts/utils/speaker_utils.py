#!/usr/bin/env python3
"""Shared helpers for transcript/protocol speaker handling.

Used by ``tag_transcript_tops.py`` and ``match_speakers.py``. Provides transcript
parsing, German name canonicalisation (lifted from ``example_data/match_speakers.py``)
and — new for the prose-style committee protocols — a speaker *directory* built from
the protocol cover (attendance list) and the body's indirect-speech name+role
mentions (``Dr. Benjamin Grimm (Minister des MdJD)``).
"""

from __future__ import annotations

import re

from preprocess_protocol import split_front_matter

# --------------------------------------------------------------------------- regexes

SPK_TAG_RE = re.compile(r"<SD-SPK>\s*(.*?)\s*</SD>")
SEG_LINE_RE = re.compile(r"^\[[0-9:.]+ --> [0-9:.]+\]\s*(.*)$")

# Leading titles stripped to get the bare person name / surname.
TITLE_TOKENS = {
    "dr.", "prof.", "habil.", "dr", "prof",
    "präsidentin", "präsident", "vizepräsidentin", "vizepräsident",
    "alterspräsident", "alterspräsidentin",
}
# Leading office/role words dropped from a *display* name (party stays in the paren).
ROLE_OFFICE_TOKENS = {
    "abgeordneter", "abgeordnete", "minister", "ministerin", "staatssekretär",
    "staatssekretärin", "staatsministerin", "staatsminister", "vorsitzender",
    "vorsitzende", "altersvorsitzender", "altersvorsitzende", "herr", "frau",
    "parlamentarischer", "parlamentarische",
}
# Lowercase particles allowed inside a surname ("André von Ossowski").
NAME_PARTICLES = {"von", "van", "de", "der", "den", "zu", "zur", "zum", "la", "le", "di", "da"}
NAME_TOKEN_RE = re.compile(r"^[A-ZÄÖÜ][\wäöüß.\-']*$")

# Give-the-floor trigger words: a chair turn must contain one before we trust a
# surname in it as "the next speaker".
CUE_TRIGGER_RE = re.compile(
    r"(das wort|hat das wort|spricht für|spricht jetzt|erteile|rufe|"
    r"bitten|\bbitte\b|am mikrofon|das wort geht an|übergeben|"
    r"fortsetzen|fort\.|setzen die aussprache)",
    re.IGNORECASE,
)
# Signals that the chair turn is NOT a clean single hand-off (queue / pronoun
# deferral) — we defer those to the LLM rather than guess.
DEFER_SIGNAL_RE = re.compile(
    r"(\bsie\b|\bihnen\b|noch mal|nochmal|auf meiner liste|wir sammeln|"
    r"sammeln wir|der reihe nach)",
    re.IGNORECASE,
)

_PARTICLE_ALT = "|".join(sorted(NAME_PARTICLES))
# A "Name (role)" unit: capitalised tokens (or particles) followed by a paren group.
NAME_PAREN_RE = re.compile(
    r"((?:[A-ZÄÖÜ][\wäöüß.\-']*|" + _PARTICLE_ALT + r")"
    r"(?:\s+(?:[A-ZÄÖÜ][\wäöüß.\-']*|" + _PARTICLE_ALT + r"))*)\s*\(([^)]{1,60})\)"
)
# Party parenthetical (committee members in the cover attendance list).
PARTY_RE = re.compile(r"\b(SPD|CDU|AfD|BSW|GRÜNE|Grüne|FDP|Linke|DIE LINKE|fraktionslos)\b")
# Office parenthetical (ministers/guests, who appear only in the body prose).
OFFICE_RE = re.compile(
    r"(?i)(minister|staatssekretär|staatssekretaer|ministerium|staatskanzlei|"
    r"landesregierung|regierung|präsident|beauftragt)")


# --------------------------------------------------------------------------- parsing

def parse_transcript(text: str) -> list[dict]:
    """Return turns ``[{label, text, text_parts, lines}]`` in order."""
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


def _split_name_role(name: str) -> tuple[str, str]:
    """Split "André von Ossowski (BSW)" -> ("André von Ossowski", "BSW")."""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return name.strip(), ""


def canonical_name(name: str) -> tuple[str, str]:
    """Return (canonical_key, surname); strips parenthetical role and leading titles."""
    bare, _role = _split_name_role(name)
    tokens = [t for t in bare.split() if t.lower() not in TITLE_TOKENS]
    if not tokens:
        tokens = bare.split()
    key = " ".join(tokens).lower()
    surname = tokens[-1].lower() if tokens else key
    return key, surname


def looks_like_person_name(name: str) -> bool:
    """True if the (paren/title-stripped) string reads like a 1–5 token person name."""
    bare, _ = _split_name_role(name)
    tokens = [t for t in bare.split() if t.lower() not in TITLE_TOKENS]
    if not (1 <= len(tokens) <= 5):
        return False
    if not NAME_TOKEN_RE.match(tokens[-1]):
        return False
    for t in tokens:
        if NAME_TOKEN_RE.match(t) or t.lower() in NAME_PARTICLES:
            continue
        return False
    return True


def clean_display_name(name: str) -> str:
    """Drop leading office words ("Abgeordneter André …" -> "André …"); keep Dr./Prof."""
    tokens = name.split()
    while tokens and tokens[0].lower() in ROLE_OFFICE_TOKENS:
        tokens.pop(0)
    return " ".join(tokens)


# --------------------------------------------------------------- speaker directory

def _given_tokens(full: str) -> set[str]:
    """Lowercased given-name tokens (everything but the surname, titles dropped)."""
    bare, _ = _split_name_role(full)
    toks = [t.lower() for t in bare.split() if t.lower() not in TITLE_TOKENS]
    return set(toks[:-1])


def _add_person(directory: dict[str, list[dict]], full: str, role: str) -> None:
    full = clean_display_name(full).strip()
    if not full or not looks_like_person_name(full):
        return
    bare, _ = _split_name_role(full)
    if len([t for t in bare.split() if t.lower() not in TITLE_TOKENS]) < 2:
        return  # need first + last name; rejects "Justizministerkonferenz", "Verschiedenes"
    key, surname = canonical_name(full)
    people = directory.setdefault(surname, [])
    given = _given_tokens(full)
    for person in people:
        pg = _given_tokens(person["full"])
        if given <= pg or pg <= given:  # same person (e.g. "Grimm" ⊆ "Benjamin Grimm")
            if len(full) > len(person["full"]):  # keep the most complete variant
                person["full"], person["key"] = full, key
            if role and not person["role"]:
                person["role"] = role
            return
    people.append({"full": full, "role": role, "key": key, "surname": surname})


def _clean_captured_name(raw: str) -> str:
    """Trim leading sentence pollution from a body capture, keeping titles + 1 given name.

    "Stellungnahme. Dr. Benjamin Grimm" / "Digitalisierung Dr. Benjamin Grimm" ->
    "Dr. Benjamin Grimm" (the surname plus titles/particles plus at most one given name).
    """
    toks = raw.split()
    if not toks:
        return ""
    kept = [toks[-1]]
    given = 0
    for tok in reversed(toks[:-1]):
        if tok.lower() in TITLE_TOKENS or tok.lower() in NAME_PARTICLES:
            kept.insert(0, tok)
        elif given < 1:
            kept.insert(0, tok)
            given += 1
        else:
            break
    return " ".join(kept)


def extract_speaker_directory(cover_text: str, body_text: str) -> dict[str, list[dict]]:
    """Build ``{surname: [{full, role, key, surname}]}`` from cover + body.

    Committee members come from the clean cover attendance list (party parenthetical,
    ≥2 tokens). Ministers/guests — present only in the body prose — are added from
    office parentheticals, with the captured name trimmed of leading pollution.
    Same-surname variants with compatible given names collapse to one entry; a surname
    with genuinely different people stays a list (callers treat that as ambiguous).
    """
    directory: dict[str, list[dict]] = {}
    for m in NAME_PAREN_RE.finditer(cover_text or ""):
        role = m.group(2).strip()
        if PARTY_RE.search(role) and "antrag" not in role.lower():
            _add_person(directory, m.group(1).strip(), role)
    for m in NAME_PAREN_RE.finditer(body_text or ""):
        role = m.group(2).strip()
        if OFFICE_RE.search(role):
            _add_person(directory, _clean_captured_name(m.group(1).strip()), role)
    return directory


def format_speaker(person: dict) -> str:
    """Render a directory entry as a transcript tag value: "Full Name (role)"."""
    return f"{person['full']} ({person['role']})" if person.get("role") else person["full"]


_VORSITZ_RE = re.compile(r"(?im)^Vorsitz\s*:")
_VORSITZ_STOP_RE = re.compile(r"(?im)^(Protokoll|Anwesend|Stellv|##\s)")


def extract_vorsitz_keys(cover_text: str, directory: dict[str, list[dict]]) -> list[str]:
    """Canonical keys of the chair(s) named under the cover's "Vorsitz:" heading, in order."""
    m = _VORSITZ_RE.search(cover_text or "")
    if not m:
        return []
    region = cover_text[m.end():]
    stop = _VORSITZ_STOP_RE.search(region)
    if stop:
        region = region[:stop.start()]
    keys: list[str] = []
    for nm in NAME_PAREN_RE.finditer(region):
        if not PARTY_RE.search(nm.group(2)):
            continue
        _, surname = canonical_name(nm.group(1))
        people = directory.get(surname)
        if people and len(people) == 1 and people[0]["key"] not in keys:
            keys.append(people[0]["key"])
    return keys
