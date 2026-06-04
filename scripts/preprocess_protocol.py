#!/usr/bin/env python3
"""Strip the non-inferable header from protocol Markdown.

A committee protocol begins with a title page, attendance list and agenda
(``Tagesordnung``) — none of which can be inferred from the transcript. The
substantive minutes start at the first ``Zu TOP 1`` heading. This script (and
its importable ``strip_before_marker``) removes everything before that marker so
the training target contains only what the model could plausibly produce.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tqdm import tqdm

DEFAULT_MARKER = r"(?i)zu\s+TOP\s*1\b"
_FRONT_MATTER_RE = re.compile(r"\A(---\n.*?\n---\n)", re.DOTALL)


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

    Front matter is preserved. Returns (result_text, matched).
    """
    front, body = split_front_matter(text)
    m = re.search(marker, body)
    if not m:
        return text, False
    return front + body[m.start():].lstrip(), True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="A protocol .md file or a directory of .md files")
    p.add_argument("--out-dir", type=Path, default=Path("data/protocols/md_clean"),
                   help="Output directory (default: data/protocols/md_clean)")
    p.add_argument("--marker", default=DEFAULT_MARKER,
                   help=f"Regex for the body-start marker (default: {DEFAULT_MARKER!r})")
    p.add_argument("--keep-original-on-no-match", action="store_true",
                   help="Write the file unchanged when the marker is absent "
                        "(default: skip it and count as a failure)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process even if output .md already exists")
    args = p.parse_args()

    inputs = iter_inputs(args.input, (".md",))
    if not inputs:
        print(f"no .md inputs found at {args.input}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for src in tqdm(inputs, desc="strip-header", unit="file"):
        out = args.out_dir / src.name
        if out.exists() and not args.overwrite:
            continue
        text = src.read_text(encoding="utf-8")
        result, matched = strip_before_marker(text, args.marker)
        if not matched:
            if args.keep_original_on_no_match:
                print(f"no marker in {src.name}; keeping original", file=sys.stderr)
            else:
                failures.append((str(src), "marker not found"))
                print(f"no marker in {src.name}; skipped", file=sys.stderr)
                continue
        out.write_text(result, encoding="utf-8")

    if failures:
        print(f"\n{len(failures)} file(s) failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
