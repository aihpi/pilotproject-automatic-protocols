#!/usr/bin/env python3
"""Stage A0 — map the raw nested corpus into the flat layout the pipeline expects.

The delivered corpus (under ``data/raw``) is organised as
``Committee/[Wahlperiode/]Session/`` folders, each holding a protocol **PDF** and
a session **MP3** with inconsistent file names. Every downstream script
(``pdf_to_markdown.py``, ``transcribe.py``, ``build_dataset.py``) instead expects
**flat** input directories and pairs a protocol to its transcript by a shared
filename stem (``HA_8_10_Protokoll`` ↔ ``HA_8_10_Transkript``).

This script walks the raw tree, derives a canonical session stem
``<ABBR>[_<WP>]_<NN>`` (committee abbreviation, optional Wahlperiode, sitting
number) and stages each session by creating **relative symlinks** into the flat
layout:

    data/protocols/pdf/<stem>_Protokoll.pdf        -> raw protocol PDF
    data/transcripts/audio/<stem>_Transkript.mp3   -> raw session MP3
        (multi-part audio -> <stem>_Transkript.pt01.mp3, .pt02.mp3, … in order)

It also writes ``data/transcripts/manifest.txt`` (staged audio paths of the
*trainable* sessions, i.e. those that have both a PDF and audio — input for
``transcribe.sbatch``) and ``data/ingest_report.tsv`` (one row per session with
its derived fields, source paths and any flags). Symlinking is idempotent
(``--overwrite`` to replace, ``--dry-run`` to preview).

After transcription, ``--merge-parts`` concatenates the per-part transcript
markdown (``<stem>_Transkript.ptNN.md``) back into one ``<stem>_Transkript.md``
so the dataset builder sees a single transcript per session.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PDF_SUFFIXES = (".pdf",)
AUDIO_SUFFIXES = (".mp3",)

# A leading "07.WP " / "8. WP" style prefix on a session-folder name (some folders
# repeat the Wahlperiode before the sitting number); stripped before reading NN.
WP_PREFIX_RE = re.compile(r"^\s*0?\d{1,2}\s*\.?\s*WP\b\.?\s*", re.IGNORECASE)
# A "N. WP" path component (the Wahlperiode layer).
WP_DIR_RE = re.compile(r"^\s*0?(\d{1,2})\s*\.?\s*WP\s*$", re.IGNORECASE)
# Committee abbreviation inside the top folder's parens, e.g. "A 7 (ASGIV, AASGZ)".
PAREN_RE = re.compile(r"\(([^)]*)\)")
FIRST_INT_RE = re.compile(r"(\d+)")


def committee_abbr(top: str) -> str:
    """Committee abbreviation from a top-level folder name.

    ``A 7 (ASGIV, AASGZ)`` -> ``ASGIV`` (first token in the parens); folders with
    no parens (``EK 82``, ``SLausitz``) -> the name with whitespace removed.
    """
    m = PAREN_RE.search(top)
    if m:
        first = re.split(r"[,/;]", m.group(1).strip())[0]
        return re.sub(r"\s+", "", first)
    return re.sub(r"\s+", "", top)


def derive_stem(rel_parts: list[str]) -> tuple[str, str, str | None, int | None]:
    """Map a session's path components (relative to the raw root) to its fields.

    Returns ``(stem, committee_abbr, wp, sitting)`` where wp/sitting may be None
    when they cannot be parsed (the caller flags those).
    """
    top = rel_parts[0]
    session = rel_parts[-1]
    abbr = committee_abbr(top)

    wp: str | None = None
    for part in rel_parts:
        m = WP_DIR_RE.match(part)
        if m:
            wp = m.group(1)
            break

    name = WP_PREFIX_RE.sub("", session)
    m = FIRST_INT_RE.search(name)
    sitting = int(m.group(1)) if m else None

    pieces = [abbr]
    if wp:
        pieces.append(wp)
    pieces.append(str(sitting) if sitting is not None else "x")
    return "_".join(pieces), abbr, wp, sitting


def pick_protocol(pdfs: list[Path]) -> Path:
    """Choose the protocol PDF when a session folder holds several.

    Prefer a full protocol (``gesamt`` / ``protokoll``) over an agenda or excerpt;
    fall back to the first by name.
    """
    def rank(p: Path) -> int:
        n = p.name.lower()
        if "gesamt" in n or "protokoll" in n:
            return 0
        if "anlage" in n:
            return 1
        return 2
    return sorted(pdfs, key=lambda p: (rank(p), p.name))[0]


def find_sessions(raw_dir: Path) -> list[Path]:
    """Every directory directly containing at least one PDF or MP3 (a session)."""
    sessions: list[Path] = []
    for d in sorted(p for p in raw_dir.rglob("*") if p.is_dir()):
        if any(c.is_file() and c.suffix.lower() in PDF_SUFFIXES + AUDIO_SUFFIXES
               for c in d.iterdir()):
            sessions.append(d)
    return sessions


def relink(link: Path, target: Path, *, overwrite: bool, dry_run: bool) -> str:
    """Create a relative symlink ``link`` -> ``target``. Returns a status word."""
    rel = os.path.relpath(target, link.parent)
    if link.is_symlink() or link.exists():
        if not overwrite:
            return "exists"
        if not dry_run:
            link.unlink()
    if dry_run:
        return "would-link"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(rel)
    return "linked"


def ingest(args: argparse.Namespace) -> int:
    raw_dir: Path = args.raw_dir
    if not raw_dir.is_dir():
        print(f"{raw_dir} is not a directory (expected the data/raw symlink)", file=sys.stderr)
        return 1

    pdf_dir = args.out_root / "protocols" / "pdf"
    audio_dir = args.out_root / "transcripts" / "audio"
    manifest_path = args.out_root / "transcripts" / "manifest.txt"
    report_path = args.out_root / "ingest_report.tsv"

    sessions = find_sessions(raw_dir)
    print(f"found {len(sessions)} session folder(s) under {raw_dir}", file=sys.stderr)

    rows: list[dict] = []
    seen: dict[str, Path] = {}
    manifest: list[str] = []

    for sess in sessions:
        rel_parts = sess.relative_to(raw_dir).parts
        stem, abbr, wp, sitting = derive_stem(list(rel_parts))
        flags: list[str] = []

        pdfs = sorted(p for p in sess.iterdir()
                      if p.is_file() and p.suffix.lower() in PDF_SUFFIXES)
        audios = sorted(p for p in sess.iterdir()
                        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)

        if sitting is None:
            flags.append("NO_SITTING_NUM")
        if stem in seen:
            flags.append(f"COLLISION_WITH:{seen[stem].relative_to(raw_dir)}")
        else:
            seen[stem] = sess
        if not pdfs:
            flags.append("MISSING_PDF")
        if not audios:
            flags.append("MISSING_AUDIO")
        if len(audios) > 1:
            flags.append(f"MULTIPART({len(audios)})")

        # Stage protocol PDF (skip on collision so we never clobber the first).
        pdf_src = None
        if pdfs and "COLLISION_WITH" not in "".join(flags):
            pdf_src = pick_protocol(pdfs)
            link = pdf_dir / f"{stem}_Protokoll.pdf"
            relink(link, pdf_src, overwrite=args.overwrite, dry_run=args.dry_run)

        # Stage audio (single -> _Transkript.mp3; multi -> _Transkript.ptNN.mp3).
        staged_audio: list[Path] = []
        if audios and "COLLISION_WITH" not in "".join(flags):
            if len(audios) == 1:
                names = [audio_dir / f"{stem}_Transkript{audios[0].suffix.lower()}"]
            else:
                names = [audio_dir / f"{stem}_Transkript.pt{i:02d}{a.suffix.lower()}"
                         for i, a in enumerate(audios, 1)]
            for link, src in zip(names, audios):
                relink(link, src, overwrite=args.overwrite, dry_run=args.dry_run)
                staged_audio.append(link)

        # Manifest = trainable sessions only (both a protocol and audio present).
        if pdf_src and staged_audio:
            manifest.extend(str(p) for p in staged_audio)

        rows.append({
            "stem": stem, "committee": abbr, "wp": wp or "-",
            "sitting": sitting if sitting is not None else "-",
            "n_pdf": len(pdfs), "n_audio": len(audios),
            "flags": ";".join(flags) if flags else "OK",
            "pdf_src": str(pdf_src.relative_to(raw_dir)) if pdf_src else "-",
            "audio_src": (str(audios[0].relative_to(raw_dir)) if audios else "-")
            + (f" (+{len(audios)-1} more)" if len(audios) > 1 else ""),
        })

    # Reports.
    cols = ["stem", "committee", "wp", "sitting", "n_pdf", "n_audio", "flags",
            "pdf_src", "audio_src"]
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for r in sorted(rows, key=lambda r: r["stem"]):
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("\n".join(manifest) + ("\n" if manifest else ""),
                                 encoding="utf-8")

    n_ok = sum(1 for r in rows if r["flags"] == "OK")
    n_flag = len(rows) - n_ok
    print(f"\n=== ingest summary ===", file=sys.stderr)
    print(f"sessions: {len(rows)} ({n_ok} clean, {n_flag} flagged)", file=sys.stderr)
    print(f"trainable (PDF + audio) sessions in manifest: "
          f"{len(set(Path(m).name.split('_Transkript')[0] for m in manifest))} "
          f"({len(manifest)} audio file(s))", file=sys.stderr)
    for r in rows:
        if r["flags"] != "OK":
            print(f"  FLAG {r['stem']}: {r['flags']}", file=sys.stderr)
    if args.dry_run:
        print("(dry-run: no symlinks, manifest or report written)", file=sys.stderr)
    else:
        print(f"wrote {report_path} and {manifest_path}", file=sys.stderr)
    return 0


def merge_parts(args: argparse.Namespace) -> int:
    """Concatenate ``<stem>_Transkript.ptNN.md`` parts into ``<stem>_Transkript.md``."""
    tdir: Path = args.transcript_dir
    if not tdir.is_dir():
        print(f"{tdir} is not a directory", file=sys.stderr)
        return 1
    part_re = re.compile(r"^(.*)_Transkript\.pt(\d+)\.md$")
    groups: dict[str, list[tuple[int, Path]]] = {}
    for p in sorted(tdir.iterdir()):
        m = part_re.match(p.name)
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), p))
    if not groups:
        print(f"no *_Transkript.ptNN.md parts found in {tdir}", file=sys.stderr)
        return 0
    for stem, parts in sorted(groups.items()):
        parts.sort()
        out = tdir / f"{stem}_Transkript.md"
        if out.exists() and not args.overwrite:
            print(f"  {out.name} exists; use --overwrite", file=sys.stderr)
            continue
        body = "\n\n".join(p.read_text(encoding="utf-8").strip() for _, p in parts)
        out.write_text(body + "\n", encoding="utf-8")
        print(f"  merged {len(parts)} part(s) -> {out.name}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                   help="Root of the nested raw corpus (default: data/raw)")
    p.add_argument("--out-root", type=Path, default=Path("data"),
                   help="Working data root to stage into (default: data)")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing symlinks / merged files")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be staged without writing anything")
    p.add_argument("--merge-parts", action="store_true",
                   help="Post-transcription: merge <stem>_Transkript.ptNN.md parts "
                        "into <stem>_Transkript.md (uses --transcript-dir)")
    p.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts/md"),
                   help="Transcript md dir for --merge-parts (default: data/transcripts/md)")
    args = p.parse_args()

    if args.merge_parts:
        return merge_parts(args)
    return ingest(args)


if __name__ == "__main__":
    sys.exit(main())
