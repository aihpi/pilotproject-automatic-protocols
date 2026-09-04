#!/usr/bin/env python3
"""Build the held-out evaluation set under ``test/``.

Run from the repo root:
    uv run python scripts/build_test_set.py --source-dir <dir>

``<dir>`` holds ``transcripts/md_prepared/<stem>_Transkript.md`` and
``protocols/md/<stem>_Protokoll.md`` for the six stems below (on the HPI cluster
that is ``archive/data_example02`` in the projects clone). The six sessions are
listed in ``test/manifest.tsv``, which ``build_dataset.py --holdout-manifest``
reads to keep them out of every training split. Only the manifest and README are
tracked; the transcripts and gold protocols are regenerated locally.

For each session:
  * gold protocol is run through ``preprocess_protocol.clean_protocol`` (same as
    the training targets) → cover page off, page-footer tables stripped;
  * transcript is copied with every unresolved diarisation tag
    ``<SD-SPK>SPEAKER_NN</SD>`` rewritten to an obviously-synthetic placeholder
    name + fraction, e.g. ``Max Mustermann (ABC)`` (deterministic, reset per
    session). Resolved real names are kept.

Consequence: rewritten turns no longer match the gold's real names — gold
name-match is N/A there (flagged per session in the manifest).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_protocol import clean_protocol  # noqa: E402

# (dir-name, stem). Stems are the file prefixes ``<stem>_Transkript.md`` /
# ``<stem>_Protokoll.md``. All six are committee sessions; the plenary Plenum_8-31
# (structurally different) is deliberately excluded.
SESSIONS = [
    ("short_ARD_1",  "ARD_1"),
    ("mid_HA_1",     "HA_1"),
    ("mid_HA_3",     "HA_3"),
    ("mid_ABJS_5",   "ABJS_5"),
    ("long_SBue_2",  "SBue_2"),
    ("long_SBue_3",  "SBue_3"),
]

OUT_DIR = Path("test")

# Obviously-synthetic placeholder identities (German "Mustermann"/"Musterfrau" =
# specimen names) + nonsense 3-letter fraction codes (never a real fraction like
# SPD/CDU/AfD/BSW). 26 distinct, enough for the most unknown-heavy session (17).
_FIRST = ["Max", "Erika", "Hans", "Petra", "Klaus", "Sabine", "Peter", "Anna",
          "Thomas", "Julia", "Frank", "Maria", "Stefan", "Claudia", "Andreas",
          "Birgit", "Michael", "Sandra", "Jürgen", "Heike", "Dirk", "Nicole",
          "Bernd", "Katrin", "Lars", "Ute"]
_CODES = ["ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STU", "VWX", "YZA", "BCD",
          "EFG", "HIJ", "KLM", "NOP", "QRS", "TUV", "WXY", "ZAB", "CDE", "FGH",
          "IJK", "LMN", "OPQ", "RST", "UVW", "XYZ"]


def _placeholder(i: int) -> str:
    first = _FIRST[i % len(_FIRST)]
    surname = "Mustermann" if i % 2 == 0 else "Musterfrau"
    return f"{first} {surname} ({_CODES[i % len(_CODES)]})"


_SPK_RE = re.compile(r"SPEAKER_\d+")


def rewrite_unknown_speakers(text: str) -> tuple[str, dict[str, str]]:
    """Replace each distinct SPEAKER_NN with a deterministic placeholder name,
    assigned in order of first appearance. Returns (new_text, mapping)."""
    mapping: dict[str, str] = {}
    for m in _SPK_RE.finditer(text):
        tok = m.group(0)
        if tok not in mapping:
            mapping[tok] = _placeholder(len(mapping))
    if mapping:
        text = _SPK_RE.sub(lambda m: mapping[m.group(0)], text)
    return text, mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True, type=Path,
                    help="directory holding transcripts/md_prepared/ and protocols/md/")
    args = ap.parse_args()
    src = str(args.source_dir)
    rows = []
    OUT_DIR.mkdir(exist_ok=True)
    for dirname, stem in SESSIONS:
        tx_src = args.source_dir / "transcripts" / "md_prepared" / f"{stem}_Transkript.md"
        gold_src = args.source_dir / "protocols" / "md" / f"{stem}_Protokoll.md"
        if not tx_src.exists() or not gold_src.exists():
            raise SystemExit(f"missing source for {stem}: {tx_src} / {gold_src}")

        dest = OUT_DIR / dirname
        dest.mkdir(parents=True, exist_ok=True)

        # transcript: rewrite unknown speakers
        raw = tx_src.read_text(encoding="utf-8")
        new_tx, mapping = rewrite_unknown_speakers(raw)
        (dest / f"{stem}_Transkript.md").write_text(new_tx, encoding="utf-8")

        # gold: strip cover + page-footer tables (same cleaning as training targets)
        _, gold_body, matched = clean_protocol(gold_src.read_text(encoding="utf-8"))
        if not matched:
            raise SystemExit(f"clean_protocol did not match for {stem} "
                             "(no 'Beschlüsse und Festlegungen' heading?)")
        gold_body = gold_body.strip() + "\n"
        (dest / f"{stem}_Protokoll.md").write_text(gold_body, encoding="utf-8")

        tops = len(re.findall(r"<SD-TOP>", raw))
        speakers = len(set(re.findall(r"<SD-SPK>([^<]*)</SD>", raw)))
        rows.append({
            "example": dirname, "stem": stem, "source": src,
            "tops": tops, "speakers": speakers,
            "invented_speakers": len(mapping),
            "transcript_words": len(raw.split()), "gold_words": len(gold_body.split()),
        })
        if mapping:
            log = "".join(f"{k}\t{v}\n" for k, v in mapping.items())
            (dest / "speaker_map.tsv").write_text(log, encoding="utf-8")
        print(f"  {dirname}: {tops} TOPs, {speakers} speakers, {len(mapping)} invented, "
              f"gold {len(gold_body.split())}w (cleaned)", flush=True)

    cols = ["example", "stem", "source", "tops", "speakers", "invented_speakers",
            "transcript_words", "gold_words", "split_note"]
    lines = ["\t".join(cols)]
    for r in rows:
        r["split_note"] = ("held out of every training split via --holdout-manifest; "
                           "gold cleaned like the training targets; invented-speaker "
                           "turns have no gold name match")
        lines.append("\t".join(str(r[c]) for c in cols))
    (OUT_DIR / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR}/manifest.tsv ({len(rows)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
