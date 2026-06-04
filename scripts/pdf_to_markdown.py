#!/usr/bin/env python3
"""Convert protocol PDFs (Ausschussprotokolle) to Markdown with Docling.

Docling does layout-aware extraction (reading order, tables, headings) and
exports clean Markdown, which is more robust on the complex multi-column
protocol layouts than a raw text dump. A small YAML front-matter header is added,
mirroring the style of ``scripts/transcribe.py``.

Documents that yield almost no text (e.g. an image-only scan with OCR disabled)
are reported as failures rather than producing empty targets.
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

# Below this many characters a conversion is treated as effectively empty.
MIN_CHARS = 50


def iter_inputs(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Return the input files for a file-or-directory ``--input`` argument."""
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in suffixes)
    return [path]


def make_converter():
    """Build a Docling converter with OCR disabled (protocols are born-digital).

    Disabling OCR avoids needing an OCR engine and is much faster; table
    structure recognition (used for the voting tables) is kept on.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    opts = PdfPipelineOptions(do_ocr=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


@lru_cache(maxsize=1)
def _default_converter():
    return make_converter()


def convert_pdf(path: Path, converter=None) -> str:
    """Convert a PDF to a Markdown string via Docling."""
    conv = converter or _default_converter()
    body = conv.convert(str(path)).document.export_to_markdown()
    return html.unescape(body)


def write_md(out_path: Path, *, source: Path, body: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "---",
        f"source: {source.name}",
        "kind: protocol",
        "converter: docling",
        f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "---",
        "",
    ]
    out_path.write_text("\n".join(header) + body + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="A protocol .pdf file or a directory of .pdf files")
    p.add_argument("--out-dir", type=Path, default=Path("data/protocols/md"),
                   help="Directory for output .md files (default: data/protocols/md)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-convert even if output .md already exists")
    args = p.parse_args()

    inputs = iter_inputs(args.input, (".pdf",))
    if not inputs:
        print(f"no .pdf inputs found at {args.input}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    converter = _default_converter()

    failures: list[tuple[str, str]] = []
    for src in tqdm(inputs, desc="pdf->md", unit="file"):
        out = args.out_dir / f"{src.stem}.md"
        if out.exists() and not args.overwrite:
            continue
        try:
            body = convert_pdf(src, converter)
            if len(body.strip()) < MIN_CHARS:
                raise ValueError(
                    f"only {len(body.strip())} chars extracted "
                    f"(likely a scanned/image PDF)"
                )
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
