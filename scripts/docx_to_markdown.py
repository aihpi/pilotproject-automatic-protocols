#!/usr/bin/env python3
"""Convert transcript .docx files to Markdown with Docling.

This is the production path for *already-clean* transcripts. Docling exports
clean Markdown; any ``<SD-TOP>…</SD>`` / ``<SD-SPK>…</SD>`` markers present in the
text are preserved by default (they are training signal) and can be removed with
``--strip-tags``.

NOTE: this assumes clean transcripts. If a dataset has mis-placed speaker
boundaries, repair them in a separate pre-processing step before this converter
runs — Docling normalises away the raw docx whitespace such a repair relies on.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from pdf_to_markdown import make_converter

# An SD tag: opening "<SD-XXX>" ... closing "</SD>" (closing has no suffix).
TAG_RE = re.compile(r"<SD-[A-Z]+>.*?</SD>", re.DOTALL)


def iter_inputs(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Return the input files for a file-or-directory ``--input`` argument."""
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in suffixes)
    return [path]


@lru_cache(maxsize=1)
def _default_converter():
    return make_converter()


def convert_docx(path: Path, converter=None, *, keep_tags: bool = True) -> str:
    """Convert a .docx to a Markdown string via Docling."""
    conv = converter or _default_converter()
    body = conv.convert(str(path)).document.export_to_markdown()
    body = html.unescape(body)  # docling escapes <SD-…> tags as &lt;SD-…&gt;
    if not keep_tags:
        body = TAG_RE.sub("", body)
    return body


def write_md(out_path: Path, *, source: Path, body: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "---",
        f"source: {source.name}",
        "kind: transcript",
        "converter: docling",
        f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "---",
        "",
    ]
    out_path.write_text("\n".join(header) + body + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="A transcript .docx file or a directory of .docx files")
    p.add_argument("--out-dir", type=Path, default=Path("data/transcripts/md"),
                   help="Directory for output .md files (default: data/transcripts/md)")
    p.add_argument("--strip-tags", action="store_true",
                   help="Drop <SD-…> markers (default: keep them as training signal)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-convert even if output .md already exists")
    args = p.parse_args()

    inputs = iter_inputs(args.input, (".docx",))
    if not inputs:
        print(f"no .docx inputs found at {args.input}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    converter = _default_converter()

    failures: list[tuple[str, str]] = []
    for src in tqdm(inputs, desc="docx->md", unit="file"):
        out = args.out_dir / f"{src.stem}.md"
        if out.exists() and not args.overwrite:
            continue
        try:
            body = convert_docx(src, converter, keep_tags=not args.strip_tags)
            if not body.strip():
                raise ValueError("no text extracted from document")
            write_md(out, source=src, body=body)
        except Exception as exc:
            failures.append((str(src), repr(exc)))
            print(f"\nERROR on {src.name}: {exc!r}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} file(s) failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
