#!/usr/bin/env python3
"""Import a flat data drop (e.g. an unzipped ``Daten/``) into the nested ``data2`` corpus.

New training data arrives as a flat folder of files named like::

    1.AHF vom 07.11.2019.mp3 / .pdf
    10. AWFK am 09.10.2020.mp3 / .pdf
    8. AWFK am 16.09.2020 Teil 1.mp3 / Teil 2.mp3   (one session, two audio parts)
    10.AHF vom 30.03.2020 (2).mp3                    (same session, a second recording)

``scripts/ingest_corpus.py`` expects the nested layout ``Committee/[N. WP/]Session/{pdf,mp3}``
and derives the session stem from the *folder path* (committee abbreviation, optional
Wahlperiode, sitting number). This script reshapes a flat drop into that layout so a later
``ingest_corpus.py`` run stages it unchanged.

Two subcommands:

``adopt``
    The shared corpus committee folders are often owned by ``root`` and not writable. Without
    sudo this relocates a committee's root-owned folder to a ``data2_archive/`` sibling (atomic
    same-filesystem rename, originals preserved and out of the scanned tree), recreates a
    user-owned folder in its place, and copies the original sessions back. Run once per
    committee before importing into it. Idempotent: a folder you already own is skipped.

``import``
    Parse each flat filename, group files into sessions, and copy them into
    ``<data2>/<committee>/<WP>. WP/<N>. Sitzung <ABBR> am <DD.MM.YYYY>/`` with one ``<N>.pdf``
    and one or more ``A<NN>-YYYY-MM-DD[.TeilK].mp3`` (multipart files sort into ingest order).
    Writes ``data/import_report.tsv``. Idempotent (skip-if-exists); ``--dry-run`` writes nothing.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the committee/stem parsing from the pipeline's ingest_corpus so both stay in
# lock-step. This script lives outside scripts/ (one-off data onboarding, not part of the
# repeatable pipeline), so point at the sibling scripts/ directory explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_corpus import FIRST_INT_RE, PAREN_RE  # noqa: E402

# <sitting>. <ABBR> {vom|am} <DD.MM.YYYY> [Teil N] [(2)] .{mp3,pdf}
DROP_RE = re.compile(
    r"^(?P<sitting>\d+)\s*\.\s*(?P<abbr>[A-Za-zÄÖÜäöüß]+)\s+(?:vom|am)\s+"
    r"(?P<day>\d{2})\.(?P<month>\d{2})\.(?P<year>\d{4})"
    r"(?:\s+Teil\s+(?P<teil>\d+))?"
    r"(?:\s+\((?P<dup>\d+)\))?"
    r"(?:-(?P<part>\d+))?"           # "…-1.mp3" / "…-2.mp3" multipart suffix
    r"\s*\.(?P<ext>mp3|pdf)$",       # tolerate a trailing space before the extension
    re.IGNORECASE,
)


@dataclass
class DropFile:
    path: Path
    abbr: str
    sitting: int
    day: str
    month: str
    year: str
    teil: int | None
    dup: int | None
    ext: str  # "mp3" or "pdf"

    @property
    def order_key(self) -> tuple[int, int, str]:
        # Plain part first, then Teil 1..N, then a "(2)" extra recording.
        return (self.teil or 0, self.dup or 0, self.path.name)


@dataclass
class Session:
    abbr: str
    sitting: int
    day: str
    month: str
    year: str
    pdfs: list[DropFile] = field(default_factory=list)
    audios: list[DropFile] = field(default_factory=list)

    @property
    def date(self) -> str:
        return f"{self.day}.{self.month}.{self.year}"


def build_committee_index(data2_root: Path) -> dict[str, Path]:
    """Map every committee abbreviation to its top-level folder in ``data2``.

    Maps *all* parenthesised tokens (so ``A 7 (ASGIV, AASGZ)`` answers both ASGIV and AASGZ).
    """
    index: dict[str, Path] = {}
    for d in sorted(p for p in data2_root.iterdir() if p.is_dir()):
        m = PAREN_RE.search(d.name)
        if m:
            for tok in re.split(r"[,/;]", m.group(1)):
                tok = re.sub(r"\s+", "", tok)
                if tok:
                    index.setdefault(tok.upper(), d)          # match abbr case-insensitively
        else:
            index.setdefault(re.sub(r"\s+", "", d.name).upper(), d)
    return index


def committee_number(folder: Path) -> str:
    """Two-digit committee number from a folder like ``A 11 (AHF)`` -> ``11`` (``A 6`` -> ``06``)."""
    m = FIRST_INT_RE.search(folder.name)
    return f"{int(m.group(1)):02d}" if m else "00"


def parse_drop_name(path: Path) -> DropFile | None:
    m = DROP_RE.match(path.name)
    if not m:
        return None
    return DropFile(
        path=path,
        abbr=m.group("abbr").upper(),
        sitting=int(m.group("sitting")),
        day=m.group("day"),
        month=m.group("month"),
        year=m.group("year"),
        teil=int(m.group("teil") or m.group("part")) if (m.group("teil") or m.group("part")) else None,
        dup=int(m.group("dup")) if m.group("dup") else None,
        ext=m.group("ext").lower(),
    )


def group_sessions(drop_dir: Path) -> tuple[dict[tuple, Session], list[Path]]:
    """Group drop files into sessions keyed by (abbr, sitting, date). Returns (sessions, unparsed)."""
    sessions: dict[tuple, Session] = {}
    unparsed: list[Path] = []
    for p in sorted(drop_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".mp3", ".pdf"):
            continue
        df = parse_drop_name(p)
        if df is None:
            unparsed.append(p)
            continue
        key = (df.abbr, df.sitting, df.year, df.month, df.day)
        sess = sessions.get(key)
        if sess is None:
            sess = Session(df.abbr, df.sitting, df.day, df.month, df.year)
            sessions[key] = sess
        (sess.pdfs if df.ext == "pdf" else sess.audios).append(df)
    return sessions, unparsed


def wp_for(session: Session, args: argparse.Namespace) -> int:
    if args.wp_from_date:
        return 7 if int(session.year) <= 2024 else 8
    return args.wp


def _same_size(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.stat().st_size == b.stat().st_size


# --------------------------------------------------------------------------- adopt


def adopt(args: argparse.Namespace) -> int:
    data2_root = args.data2_root.resolve()
    archive_root = args.archive_root or (data2_root.parent / "data2_archive")
    uid = os.getuid()
    rc = 0
    for name in args.committees:
        src = data2_root / name
        if not src.is_dir():
            print(f"  SKIP {name}: not found under {data2_root}", file=sys.stderr)
            rc = 1
            continue
        if src.stat().st_uid == uid and os.access(src, os.W_OK):
            print(f"  OK   {name}: already user-owned and writable", file=sys.stderr)
            continue
        archived = archive_root / name
        n_files = sum(1 for _ in src.rglob("*") if _.is_file())
        print(f"  ADOPT {name}: {n_files} file(s) -> archive then copy back", file=sys.stderr)
        if args.dry_run:
            continue
        archive_root.mkdir(parents=True, exist_ok=True)
        if archived.exists():
            print(f"    archive target {archived} already exists; refusing to overwrite",
                  file=sys.stderr)
            rc = 1
            continue
        shutil.move(str(src), str(archived))          # atomic rename on the same filesystem
        src.mkdir(parents=True)                        # fresh, user-owned
        for child in sorted(archived.iterdir()):       # copy originals back (user-owned copies)
            dst = src / child.name
            if child.is_dir():
                shutil.copytree(child, dst)
            else:
                shutil.copy2(child, dst)
        new_files = sum(1 for _ in src.rglob("*") if _.is_file())
        if new_files != n_files:
            print(f"    WARNING count mismatch: archive {n_files} != copy {new_files}",
                  file=sys.stderr)
            rc = 1
        else:
            print(f"    verified {new_files} file(s); original preserved at {archived}",
                  file=sys.stderr)
    return rc


# -------------------------------------------------------------------------- import


def import_drop(args: argparse.Namespace) -> int:
    drop_dir: Path = args.drop_dir
    if not drop_dir.is_dir():
        print(f"{drop_dir} is not a directory", file=sys.stderr)
        return 1
    data2_root = args.data2_root.resolve()
    index = build_committee_index(data2_root)
    sessions, unparsed = group_sessions(drop_dir)

    for p in unparsed:
        print(f"  WARN unparsable, skipped: {p.name}", file=sys.stderr)

    rows: list[dict] = []
    rc = 0
    for key in sorted(sessions):
        sess = sessions[key]
        flags: list[str] = []
        committee_folder = index.get(sess.abbr)
        if committee_folder is None:
            print(f"  ERROR unknown committee {sess.abbr!r} (session {key}); skipped",
                  file=sys.stderr)
            rc = 1
            continue
        wp = wp_for(sess, args)
        cnum = committee_number(committee_folder)
        dest_dir = (committee_folder / f"{wp}. WP" /
                    f"{sess.sitting}. Sitzung {sess.abbr} am {sess.date}")

        # PDF: dedupe by byte-size; keep one. Differing sizes -> keep first, flag.
        pdf_srcs = [d.path for d in sorted(sess.pdfs, key=lambda d: d.order_key)]
        chosen_pdf = pdf_srcs[0] if pdf_srcs else None
        if len(pdf_srcs) > 1:
            sizes = {p.stat().st_size for p in pdf_srcs}
            flags.append("DUP_PDF_SAME" if len(sizes) == 1 else "MULTI_PDF_DIFFER")
        if not pdf_srcs:
            flags.append("MISSING_PDF")

        # Audio: ordered parts. A "(2)" extra recording becomes another part.
        audio_srcs = [d for d in sorted(sess.audios, key=lambda d: d.order_key)]
        if not audio_srcs:
            flags.append("MISSING_AUDIO")
        if any(d.dup for d in audio_srcs):
            flags.append("DUP_AS_MULTIPART")
        if len(audio_srcs) > 1 and not any(d.dup for d in audio_srcs):
            flags.append(f"MULTIPART({len(audio_srcs)})")

        date_iso = f"{sess.year}-{sess.month}-{sess.day}"
        if len(audio_srcs) == 1:
            audio_names = [f"A{cnum}-{date_iso}.mp3"]
        else:
            audio_names = [f"A{cnum}-{date_iso}.Teil{i}.mp3"
                           for i in range(1, len(audio_srcs) + 1)]

        # Plan the copy actions: (src, dst).
        actions: list[tuple[Path, Path]] = []
        if chosen_pdf is not None:
            actions.append((chosen_pdf, dest_dir / f"{sess.sitting}.pdf"))
        for df, name in zip(audio_srcs, audio_names):
            actions.append((df.path, dest_dir / name))

        # Execute.
        status = "planned"
        if not args.dry_run:
            if not os.access(committee_folder, os.W_OK):
                print(f"  ERROR {committee_folder} not writable; run the 'adopt' subcommand first",
                      file=sys.stderr)
                return 1
            dest_dir.mkdir(parents=True, exist_ok=True)
            done, skipped = 0, 0
            for src, dst in actions:
                if dst.exists() and not args.overwrite:
                    if not _same_size(src, dst):
                        flags.append(f"EXISTS_DIFFERS:{dst.name}")
                    skipped += 1
                    continue
                shutil.copy2(src, dst)
                done += 1
            status = f"copied:{done} skipped:{skipped}"

        stem = "_".join([sess.abbr, str(wp), str(sess.sitting)])
        rows.append({
            "target_stem": stem,
            "committee": sess.abbr,
            "wp": wp,
            "sitting": sess.sitting,
            "date": sess.date,
            "dest_dir": str(dest_dir.relative_to(data2_root)),
            "n_pdf": len(pdf_srcs),
            "n_audio": len(audio_srcs),
            "op": "copy" if not args.dry_run else "dry-run",
            "status": status,
            "flags": ";".join(flags) if flags else "OK",
            "src_files": "; ".join(p.name for p in
                                   ([chosen_pdf] if chosen_pdf else []) +
                                   [d.path for d in audio_srcs]),
        })
        print(f"  {stem}: {len(pdf_srcs)} pdf, {len(audio_srcs)} audio -> {dest_dir.name}"
              f"  [{rows[-1]['flags']}]", file=sys.stderr)

    cols = ["target_stem", "committee", "wp", "sitting", "date", "dest_dir",
            "n_pdf", "n_audio", "op", "status", "flags", "src_files"]
    if not args.dry_run:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for r in sorted(rows, key=lambda r: r["target_stem"]):
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print(f"\nwrote {args.report} ({len(rows)} session(s))", file=sys.stderr)
    else:
        print(f"\n(dry-run: nothing written; {len(rows)} session(s) planned)", file=sys.stderr)
    if unparsed:
        print(f"{len(unparsed)} file(s) could not be parsed", file=sys.stderr)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("adopt", help="Take ownership of root-owned committee folders (no sudo).")
    pa.add_argument("--committees", nargs="+", required=True,
                    help='Committee folder names, e.g. "A 11 (AHF)" "A 6 (AWFK)"')
    pa.add_argument("--data2-root", type=Path, default=Path("data/raw"),
                    help="Corpus root (default: data/raw symlink)")
    pa.add_argument("--archive-root", type=Path, default=None,
                    help="Where to move originals (default: <data2>/../data2_archive)")
    pa.add_argument("--dry-run", action="store_true")
    pa.set_defaults(func=adopt)

    pi = sub.add_parser("import", help="Reshape a flat drop into the nested corpus.")
    pi.add_argument("--drop-dir", type=Path, required=True,
                    help="Flat unzipped drop directory (e.g. .../Daten)")
    pi.add_argument("--data2-root", type=Path, default=Path("data/raw"),
                    help="Corpus root (default: data/raw symlink)")
    pi.add_argument("--wp", type=int, default=7, help="Wahlperiode for the drop (default: 7)")
    pi.add_argument("--wp-from-date", action="store_true",
                    help="Derive WP from the session year (<=2024 -> 7, else 8)")
    pi.add_argument("--report", type=Path, default=Path("data/import_report.tsv"))
    pi.add_argument("--overwrite", action="store_true", help="Replace existing destination files")
    pi.add_argument("--dry-run", action="store_true", help="Print the plan; write nothing")
    pi.set_defaults(func=import_drop)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
