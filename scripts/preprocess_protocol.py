#!/usr/bin/env python3
"""Clean committee protocol Markdown into training-ready bodies.

A committee protocol begins with a title page, attendance list and agenda
(``Tagesordnung``) — none of which can be inferred from the transcript. The
substantive minutes start at the ``## Beschlüsse und Festlegungen:`` heading and
run through ``## Aus der Beratung:``; everything from ``## Anlage/n:`` onward is
attachment material. This module splits off the cover, drops the attachments and
strips page-footer tables, image placeholders, hyperlinks and attachment
footnotes so the training target contains only inferable prose.

``clean_protocol`` is the importable entry point (used by ``build_dataset``); the
CLI writes one cleaned body per input plus the cover into a ``cover/`` subfolder.
The older ``strip_before_marker`` (marker-based header strip) is kept unchanged
for the plenary pipeline that still imports it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tqdm import tqdm

DEFAULT_MARKER = r"(?i)zu\s+TOP\s*1\b"
_FRONT_MATTER_RE = re.compile(r"\A(---\n.*?\n---\n)", re.DOTALL)

# Body starts here; everything before is cover/attendance/agenda.
_BODY_START_RE = re.compile(r"(?im)^##\s*Beschlüsse und Festlegungen")
# Attachments run from the first Anlage heading to EOF.
_ANLAGE_RE = re.compile(r"(?im)^##\s*Anlage")
# Running page-footer table: "| Landtag Brandenburg | … | S. N |" + 2 more rows.
_PAGE_TABLE_RE = re.compile(
    r"(?m)^\|[^\n]*Landtag Brandenburg[^\n]*S\.\s*\d+[^\n]*\|\n"
    r"\|[-:\s|]+\|\n"
    r"\|[^\n]*\|[ \t]*\n?"
)
# Confirmation footer appended once the committee ratifies the protocol — not
# inferable from the transcript, must never be a training target.
_CONFIRM_FOOTER_RE = re.compile(
    r"(?m)^[ \t]*\(Dieses\s+Protokoll\s+wurde\s+durch\b[^\n]*?§\s?83[^\n]*?best[äa]tigt\.\)[ \t]*$\n?")
# Plain-text page footer (PDF→md conversion dropped the table pipes, so
# _PAGE_TABLE_RE misses it): an optional committee-abbreviation line (e.g. "HA",
# "ABJS", "P-ABJS"), the "N. (öffentliche/nichtöffentliche/Sonder-) Sitzung …"
# line, and an optional trailing date line. Anchored on the unambiguous Sitzung
# line so it never eats prose.
_PLAIN_FOOTER_RE = re.compile(
    r"(?m)"
    r"(?:^[ \t]*#{0,6}[ \t]*[A-ZÄÖÜ][A-ZÄÖÜ0-9/-]{1,9}[ \t]*\n\s*)?"
    r"^[ \t]*#{0,6}[ \t]*\d{1,3}\.\s*\((?:öffentliche|nichtöffentliche|Sonder-)[^\n]*Sitzung[^\n]*\n"
    r"(?:[ \t]*\n)*"
    r"(?:^[ \t]*#{0,6}[ \t]*\d{1,2}\.\d{1,2}\.\d{4}[ \t]*\n)?")
# TOP heading mis-rendered as a list item ("- Zu TOP 8") → proper "## Zu TOP 8".
_TOP_BULLET_RE = re.compile(r"(?m)^[ \t]*[-*][ \t]*(Zu\s+TOP\b[^\n]*)$")
# Committee-code page header/footer left as a standalone line ("HA", "## RA", "SBü",
# "EK82", "P-ABJS") or dangling at the end of a wrapped prose line (a page break merged
# it in). Shape: 2–6 uppercase letters + optional trailing "ü"/digits — excludes
# Title-case words (e.g. "Der", "Aus"), which have only the first letter capitalised.
_ABBR_CODE = r"(?:P-)?[A-ZÄÖÜ]{2,6}(?:ü|[0-9]{1,3})?"
_STANDALONE_ABBR_RE = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}|[-*])?[ \t]*(" + _ABBR_CODE + r")\.?[ \t]*$\n?")
_IMAGE_RE = re.compile(r"(?m)^[ \t]*<!-- image -->[ \t]*\n?")
_HYPERLINK_RE = re.compile(r"\[([^\]]*)\]\((?:https?:|mailto:)[^)]*\)")
# Footnote definition lines that reference an attachment (the lookahead keeps
# this from eating ordinary numbered/measurement lines).
_FOOTNOTE_DEF_RE = re.compile(r"(?m)^\d{1,3}\s+(?=.*(?:Vgl\.|Anlage)).*$")
# Trailing superscript footnote marker after sentence-ending punctuation.
_FOOTNOTE_MARK_RE = re.compile(r"(?m)([.!?])[ \t]+\d{1,3}[ \t]*$")
_INLINE_ANLAGE_RE = re.compile(r"[ \t]*\(Anlage\)")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def iter_inputs(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in suffixes)
    return [path]


def split_front_matter(text: str) -> tuple[str, str]:
    """Return (front_matter, body); front_matter is '' when absent."""
    m = _FRONT_MATTER_RE.match(text)
    if m:
        return m.group(1), text[m.end():]
    return "", text


def strip_before_marker(text: str, marker: str = DEFAULT_MARKER) -> tuple[str, bool]:
    """Drop everything before the first ``marker`` match in the body.

    Front matter is preserved. Returns (result_text, matched). Kept for the
    plenary pipeline; committee cleaning uses ``clean_protocol`` instead.
    """
    front, body = split_front_matter(text)
    m = re.search(marker, body)
    if not m:
        return text, False
    return front + body[m.start():].lstrip(), True


def split_cover(text: str) -> tuple[str, str, bool]:
    """Split a protocol into (cover, body, matched) at the body-start heading.

    ``cover`` is the front matter plus everything before the first
    ``## Beschlüsse und Festlegungen`` heading; ``body`` is that heading onward.
    ``matched`` is False (and ``body`` empty) when the heading is absent.
    """
    front, rest = split_front_matter(text)
    m = _BODY_START_RE.search(rest)
    if not m:
        return text, "", False
    return front + rest[:m.start()], rest[m.start():], True


def remove_attachments(body: str) -> str:
    """Drop everything from the first ``## Anlage`` heading to the end."""
    m = _ANLAGE_RE.search(body)
    return body[:m.start()] if m else body


def strip_page_tables(text: str) -> str:
    return _PAGE_TABLE_RE.sub("", text)


def strip_plain_footers(text: str) -> str:
    """Drop plain-text page footers the table regex misses (committee abbr +
    'N. (…) Sitzung …' + optional date), wherever a page break left them."""
    return _PLAIN_FOOTER_RE.sub("", text)


def strip_confirmation_footer(text: str) -> str:
    """Drop the post-ratification '(Dieses Protokoll wurde … § 83 … bestätigt.)'."""
    return _CONFIRM_FOOTER_RE.sub("", text)


def normalize_top_headings(text: str) -> str:
    """Rewrite a TOP heading mis-rendered as a list item ('- Zu TOP 8') to '## Zu TOP 8'."""
    return _TOP_BULLET_RE.sub(r"## \1", text)


def strip_footer_abbrevs(text: str) -> str:
    """Drop the committee-code page header/footer left as a standalone line, and the same
    code dangling at the end of a wrapped prose line (a page break merged it in). Codes are
    derived from the standalone occurrences in this document, so only its own footer code
    is stripped from line ends."""
    codes = set(_STANDALONE_ABBR_RE.findall(text))
    text = _STANDALONE_ABBR_RE.sub("", text)
    for c in sorted(codes, key=len, reverse=True):
        text = re.sub(r"[ \t]+" + re.escape(c) + r"\.?[ \t]*$", "", text, flags=re.M)
    return text


def strip_images(text: str) -> str:
    return _IMAGE_RE.sub("", text)


def strip_hyperlinks(text: str) -> str:
    """Replace ``[text](url)`` with its display text; drop it when empty."""
    return _HYPERLINK_RE.sub(lambda m: m.group(1), text)


def strip_attachment_refs(text: str) -> str:
    text = _FOOTNOTE_DEF_RE.sub("", text)
    text = _FOOTNOTE_MARK_RE.sub(r"\1", text)
    text = _INLINE_ANLAGE_RE.sub("", text)
    return text


def collapse_blanks(text: str) -> str:
    return _MULTI_BLANK_RE.sub("\n\n", text).strip() + "\n"


def has_page_table(text: str) -> bool:
    return _PAGE_TABLE_RE.search(text) is not None


def clean_protocol(text: str) -> tuple[str, str, bool]:
    """Return (cover, clean_body, matched) for a committee protocol.

    Idempotent: re-running on an already-cleaned body yields cover='' and the
    same body (no Anlage/tables/images remain to remove).
    """
    cover, body, matched = split_cover(text)
    if not matched:
        return cover, "", False
    body = remove_attachments(body)
    body = strip_page_tables(body)
    body = strip_plain_footers(body)
    body = strip_confirmation_footer(body)
    body = normalize_top_headings(body)
    body = strip_footer_abbrevs(body)
    body = strip_images(body)
    body = strip_hyperlinks(body)
    body = strip_attachment_refs(body)
    return cover, collapse_blanks(body), True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="A protocol .md file or a directory of .md files")
    p.add_argument("--out-dir", type=Path, default=Path("data/protocols/md_clean"),
                   help="Output directory for cleaned bodies (default: data/protocols/md_clean)")
    p.add_argument("--cover-dir", type=Path, default=None,
                   help="Directory for cover files (default: <out-dir>/cover)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process even if output .md already exists")
    args = p.parse_args()

    inputs = iter_inputs(args.input, (".md",))
    if not inputs:
        print(f"no .md inputs found at {args.input}", file=sys.stderr)
        return 1

    cover_dir = args.cover_dir or (args.out_dir / "cover")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cover_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    warnings = 0
    for src in tqdm(inputs, desc="clean-protocol", unit="file"):
        out = args.out_dir / src.name
        if out.exists() and not args.overwrite:
            continue
        text = src.read_text(encoding="utf-8")
        cover, body, matched = clean_protocol(text)
        if not matched:
            failures.append((str(src), "no 'Beschlüsse und Festlegungen' heading"))
            print(f"FAIL {src.name}: cover end not found "
                  f"(no 'Beschlüsse und Festlegungen' heading)", file=sys.stderr)
            continue
        # Verify a page-footer table precedes the body — confirms we cut at the
        # genuine cover/body boundary rather than a stray early heading.
        if has_page_table(cover):
            print(f"OK   {src.name}: cover end found, page-footer table present",
                  file=sys.stderr)
        else:
            warnings += 1
            print(f"WARN {src.name}: cover end found but no page-footer table in "
                  f"cover — please check the boundary", file=sys.stderr)
        out.write_text(body, encoding="utf-8")
        (cover_dir / f"{src.stem}_cover.md").write_text(cover, encoding="utf-8")

    n = len(inputs)
    print(f"\n{n - len(failures)}/{n} cleaned ({warnings} warning(s), "
          f"{len(failures)} failure(s))", file=sys.stderr)
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
