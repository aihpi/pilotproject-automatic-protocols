#!/usr/bin/env python3
"""Convert example plenary protocol PDFs to clean Markdown.

Reuses the main pipeline's Docling conversion (``pdf_to_markdown.convert_pdf``)
and header strip (``preprocess_protocol.strip_before_marker``), then applies two
example-only cleanups so the demo target text is self-contained:

* ``[text](url)`` markdown links -> ``text`` (drop hyperlinks).
* ``Drucksache 8/1234`` references -> removed (they point at documents the
  transcript can't reference and add noise to the small example set).

Output lands in ``data_example/protocols/md_clean`` (front matter compatible with
``pdf_to_markdown``), ready for ``build_dataset.py``. Skips existing outputs
unless ``--overwrite``; exit codes 0/1/2.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tqdm import tqdm

# Make the sibling pipeline scripts importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_to_markdown import convert_pdf, make_converter, write_md  # noqa: E402
from preprocess_protocol import strip_before_marker  # noqa: E402

MIN_CHARS = 50

# Plenary protocols open the verbatim record with "Beginn der Sitzung:"; everything
# before it (title page + "Inhalt" table of contents) is non-inferable header.
# (Committee protocols use "Zu TOP 1" instead — see preprocess_protocol.DEFAULT_MARKER.)
PLENARY_MARKER = r"(?i)Beginn der Sitzung"

MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# "Drucksache 8/1234", "Drucksache 8/1234-B", the plural "Drucksachen 8/1 und 8/2",
# optionally wrapped in parentheses.
DRUCKSACHE_RE = re.compile(
    r"\s*\(?\bDrucksachen?\s+\d+/\d+[A-Za-z-]*"
    r"(?:\s*(?:,|und)\s*\d+/\d+[A-Za-z-]*)*\)?"
)
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
BLANKLINES_RE = re.compile(r"\n{3,}")


def clean_example_protocol(md: str) -> str:
    """Drop hyperlinks and Drucksache references; tidy the whitespace left behind."""
    md = MD_LINK_RE.sub(r"\1", md)
    md = DRUCKSACHE_RE.sub("", md)
    # Tidy ", ," / " ," style leftovers and collapsed spacing.
    md = re.sub(r"\(\s*\)", "", md)
    md = re.sub(r"\s+([,.;:])", r"\1", md)
    md = MULTISPACE_RE.sub(" ", md)
    md = BLANKLINES_RE.sub("\n\n", md)
    return md


def iter_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() == ".pdf")
    return [path]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path("data_example/protocols/pdf"),
                   help="A protocol .pdf file or a directory of .pdf files")
    p.add_argument("--out-dir", type=Path,
                   default=Path("data_example/protocols/md_clean"),
                   help="Output directory (default: data_example/protocols/md_clean)")
    p.add_argument("--marker", default=PLENARY_MARKER,
                   help=f"Body-start marker regex (default: {PLENARY_MARKER!r})")
    p.add_argument("--keep-original-on-no-match", action="store_true",
                   help="Keep the full body when the marker is absent "
                        "(default: skip and count as a failure)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-convert even if output .md already exists")
    args = p.parse_args()

    inputs = iter_inputs(args.input)
    if not inputs:
        print(f"no .pdf inputs found at {args.input}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    converter = make_converter()

    failures: list[tuple[str, str]] = []
    for src in tqdm(inputs, desc="pdf->md_clean", unit="file"):
        out = args.out_dir / f"{src.stem}.md"
        if out.exists() and not args.overwrite:
            continue
        try:
            body = convert_pdf(src, converter)
            if len(body.strip()) < MIN_CHARS:
                raise ValueError(
                    f"only {len(body.strip())} chars extracted "
                    f"(likely a scanned/image PDF)")
            body = clean_example_protocol(body)
            stripped, matched = strip_before_marker(body, args.marker)
            if not matched:
                if args.keep_original_on_no_match:
                    print(f"no marker in {src.name}; keeping full body",
                          file=sys.stderr)
                else:
                    failures.append((str(src), "marker not found"))
                    print(f"no marker in {src.name}; skipped", file=sys.stderr)
                    continue
                stripped = body
            write_md(out, source=src, body=stripped)
        except Exception as exc:
            failures.append((str(src), repr(exc)))
            print(f"\nERROR on {src.name}: {exc!r}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} file(s) failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
