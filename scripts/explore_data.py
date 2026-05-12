"""Build the join between Mediathek recordings and Ausschussprotokolle.

Reads:
    data/Landtag-Brandenburg-Archiv/exportWP8.xml
    data/Landtag-Brandenburg-Protokolle/Ausschussprotokolle/**/*.pdf
    data/Landtag-Brandenburg-Protokolle/Tondateien/Tondateien 8. Wahlperiode/**/*.nsv
    data/Landtag-Brandenburg-Protokolle/Mediathek/*.mp4

Writes:
    tmp/mediathek_to_protocol.csv
    tmp/mediathek_protocol_mapping.md

Scope: 8. Wahlperiode only. Per user choice, when a Mediathek date matches
several committee sessions, every candidate is listed; no single match is
chosen. Inconsistencies (no-APr dates, plenum collisions, sparse PDF
numbering, stray Tondateien names, undocumented Mediathek room IDs) are
called out explicitly.
"""

from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
XML_PATH = DATA / "Landtag-Brandenburg-Archiv" / "exportWP8.xml"
PROTO_ROOT = DATA / "Landtag-Brandenburg-Protokolle"
APR_ROOT = PROTO_ROOT / "Ausschussprotokolle" / "Ausschussprotokolle 8. Wahlperiode (bis 2025-06-30)"
UPDATE_ROOT = PROTO_ROOT / "Ausschussprotokolle" / "update"
AUDIO_ROOT = PROTO_ROOT / "Tondateien" / "Tondateien 8. Wahlperiode"
MEDIATHEK = PROTO_ROOT / "Mediathek"
OUT_DIR = REPO_ROOT / "tmp"
CSV_OUT = OUT_DIR / "mediathek_to_protocol.csv"
MD_OUT = OUT_DIR / "mediathek_protocol_mapping.md"

MEDIATHEK_RE = re.compile(
    r"record_(?P<room>\d+)_(?P<channel>\d+)_start_"
    r"(?P<d>\d{4}-\d{2}-\d{2})-(?P<t>\d{2}-\d{2}-\d{2})\.mp4$"
)
APR_URL_RE = re.compile(r"parladoku/w8/apr/([^/]+)/(\d+)\.pdf")
UPDATE_NAME_RE = re.compile(
    r"^(?P<abbrev>\S+)\s+(?P<sess>\d+)\.?\s*Sitzung.*?vom\s+(?P<d>\d{2}\.\d{2}\.\d{2,4})",
    re.IGNORECASE,
)
AUDIO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_xml() -> tuple[dict, dict, set]:
    """Return (apr_by_date, code_to_name, plenum_dates).

    apr_by_date[date] -> list of {code, name, session, pdf_relpath}
    code_to_name[code] -> committee name (most common Urheber for that code)
    plenum_dates: set of dates with <DokArt>PlPr</DokArt>
    """
    apr_by_date: dict[date, list[dict]] = defaultdict(list)
    code_to_names: dict[str, list[str]] = defaultdict(list)
    plenum_dates: set[date] = set()
    seen_apr: set[tuple[str, int, date]] = set()

    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    for dok in root.iter("Dokument"):
        dok_art = (dok.findtext("DokArt") or "").strip()
        dok_dat = (dok.findtext("DokDat") or "").strip()
        if not dok_dat:
            continue
        try:
            d = datetime.strptime(dok_dat, "%d.%m.%Y").date()
        except ValueError:
            continue
        if dok_art == "PlPr":
            plenum_dates.add(d)
            continue
        if dok_art != "APr":
            continue
        urheber = (dok.findtext("Urheber") or "").strip()
        code: str | None = None
        pdf_rel: str | None = None
        session: int | None = None
        for lok in dok.findall("LokURL"):
            url = (lok.text or "").strip()
            m = APR_URL_RE.search(url)
            if m:
                code = m.group(1)
                session = int(m.group(2))
                pdf_rel = f"w8/apr/{code}/{session}.pdf"
                break
        if code is None or session is None:
            continue
        key = (code, session, d)
        if key in seen_apr:
            continue
        seen_apr.add(key)
        apr_by_date[d].append(
            {"code": code, "name": urheber, "session": session, "pdf_relpath": pdf_rel}
        )
        if urheber:
            code_to_names[code].append(urheber)

    code_to_name = {}
    for code, names in code_to_names.items():
        # most frequent Urheber wins (filters out occasional cross-listings)
        code_to_name[code] = max(set(names), key=names.count)
    return apr_by_date, code_to_name, plenum_dates


def index_local_protocols() -> tuple[dict, dict, list]:
    """Return (numbered_pdfs, update_pdfs, update_unparsed).

    numbered_pdfs[(committee_name, session_num)] -> Path
    update_pdfs[(abbrev, date)] -> Path
    update_unparsed -> list of Paths whose names didn't match the pattern
    """
    numbered: dict[tuple[str, int], Path] = {}
    for committee_dir in sorted(p for p in APR_ROOT.iterdir() if p.is_dir()):
        for pdf in sorted(committee_dir.glob("*.pdf")):
            stem = pdf.stem
            if stem.isdigit():
                numbered[(committee_dir.name, int(stem))] = pdf

    update_pdfs: dict[tuple[str, date], Path] = {}
    update_unparsed: list[Path] = []
    if UPDATE_ROOT.is_dir():
        for pdf in sorted(UPDATE_ROOT.glob("*.pdf")):
            m = UPDATE_NAME_RE.match(pdf.name)
            if not m:
                update_unparsed.append(pdf)
                continue
            d_raw = m.group("d")
            for fmt in ("%d.%m.%Y", "%d.%m.%y"):
                try:
                    d = datetime.strptime(d_raw, fmt).date()
                    break
                except ValueError:
                    d = None
            if d is None:
                update_unparsed.append(pdf)
                continue
            update_pdfs[(m.group("abbrev"), d)] = pdf
    return numbered, update_pdfs, update_unparsed


def index_audio() -> tuple[dict, list, list, list]:
    """Return (audio_by_date, stray_names, mp3_files, sequence_meetings).

    audio_by_date[date] -> list of {committee, path, prefix}
    stray_names -> list of Paths whose stems didn't fit <CODE>-<date>.nsv exactly
    mp3_files -> list of companion .mp3 Paths
    sequence_meetings -> list of {committee_folder, code, title, room,
                                  start_dt, end_dt, path} from .xml Sequence files
    """
    audio_by_date: dict[date, list[dict]] = defaultdict(list)
    stray: list[Path] = []
    mp3_files: list[Path] = []
    seq_meetings: list[dict] = []
    for committee_dir in sorted(p for p in AUDIO_ROOT.iterdir() if p.is_dir()):
        for nsv in sorted(committee_dir.glob("*.nsv")):
            m = AUDIO_DATE_RE.search(nsv.stem)
            if not m:
                stray.append(nsv)
                continue
            d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            prefix = nsv.stem.split("-")[0] if "-" in nsv.stem else ""
            clean = bool(re.fullmatch(r"[A-Za-zÄÖÜäöüß0-9]+-\d{4}-\d{2}-\d{2}", nsv.stem))
            if not clean:
                stray.append(nsv)
            audio_by_date[d].append(
                {"committee": committee_dir.name, "path": nsv, "prefix": prefix}
            )
        mp3_files.extend(sorted(committee_dir.glob("*.mp3")))
        for xml_path in sorted(committee_dir.glob("*.xml")):
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                continue
            def t(tag: str) -> str:
                el = root.find(tag)
                return (el.text or "").strip() if el is not None and el.text else ""
            start_dt = end_dt = None
            try:
                start_dt = datetime.strptime(t("MeetingStart"), "%d.%m.%Y %H:%M:%S")
            except ValueError:
                pass
            try:
                end_dt = datetime.strptime(t("MeetingEnd"), "%d.%m.%Y %H:%M:%S")
            except ValueError:
                pass
            seq_meetings.append(
                {
                    "committee_folder": committee_dir.name,
                    "code": t("MeetingCommission"),
                    "title": t("MeetingTitle"),
                    "room": t("MeetingRoom"),
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                    "path": xml_path,
                }
            )
    return audio_by_date, stray, mp3_files, seq_meetings


def index_mediathek() -> list[dict]:
    records = []
    for mp4 in sorted(MEDIATHEK.glob("*.mp4")):
        m = MEDIATHEK_RE.match(mp4.name)
        if not m:
            continue
        d = datetime.strptime(m.group("d"), "%Y-%m-%d").date()
        t = m.group("t").replace("-", ":")
        records.append(
            {
                "path": mp4,
                "room": m.group("room"),
                "channel": m.group("channel"),
                "date": d,
                "time": t,
                "start_iso": f"{d.isoformat()}T{t}",
            }
        )
    return records


def resolve_local_pdf(
    apr_entry: dict, numbered: dict, update: dict, code_to_name: dict
) -> Path | None:
    name = code_to_name.get(apr_entry["code"], apr_entry["name"])
    p = numbered.get((name, apr_entry["session"]))
    if p:
        return p
    # try update by (abbrev=code, date) — code in XML often matches the update abbrev
    return None  # update files keyed by abbrev+date, looked up separately at row time


def build_rows(
    media: list[dict],
    apr_by_date: dict,
    audio_by_date: dict,
    plenum_dates: set,
    numbered: dict,
    update: dict,
    code_to_name: dict,
) -> list[dict]:
    rows = []
    for rec in media:
        d = rec["date"]
        cand_apr = apr_by_date.get(d, [])
        cand_audio = audio_by_date.get(d, [])
        notes_parts = []
        if not cand_apr and d in plenum_dates:
            notes_parts.append("plenum day (PlPr); no APr")
        elif not cand_apr:
            notes_parts.append("no APr that date; possibly Plenum/press/other")
        elif d in plenum_dates:
            notes_parts.append("plenum day + APr same date")
            notes_parts.append(f"{len(cand_apr)} APr candidates")
        elif len(cand_apr) == 1:
            notes_parts.append("unique APr candidate")
        else:
            notes_parts.append(f"{len(cand_apr)} APr candidates")
        if cand_audio:
            notes_parts.append(f"{len(cand_audio)} audio file(s)")
        else:
            notes_parts.append("no audio that date")

        committees = []
        pdfs = []
        for e in cand_apr:
            committee_name = code_to_name.get(e["code"], e["name"])
            committees.append(f"{committee_name} ({e['code']} session {e['session']})")
            # try numbered local PDF
            p = numbered.get((committee_name, e["session"]))
            if p is None:
                # try update PDF by (abbrev=code, date)
                p = update.get((e["code"], d))
                # also try a few common abbreviation aliases
                if p is None:
                    for alias in [e["code"], e["code"].replace("SBue", "SBü")]:
                        p = update.get((alias, d))
                        if p:
                            break
            pdfs.append(str(p.relative_to(REPO_ROOT)) if p else f"MISSING: {e['pdf_relpath']}")
        audios = [
            str(a["path"].relative_to(REPO_ROOT)) for a in cand_audio
        ]
        rows.append(
            {
                "mediathek_file": rec["path"].name,
                "room_id": rec["room"],
                "start_datetime": rec["start_iso"],
                "candidate_committees": " | ".join(committees),
                "candidate_protocol_pdfs": " | ".join(pdfs),
                "candidate_audio_files": " | ".join(audios),
                "notes": "; ".join(notes_parts),
            }
        )
    return rows


def find_xml_pdfs_missing_on_disk(apr_by_date, numbered, code_to_name) -> list[str]:
    """List (code, session, date) referenced by XML but absent on disk."""
    missing = []
    for d, entries in apr_by_date.items():
        for e in entries:
            name = code_to_name.get(e["code"], e["name"])
            if (name, e["session"]) not in numbered:
                missing.append(f"{e['code']} session {e['session']} ({d.isoformat()}) → expected {e['pdf_relpath']}")
    return sorted(missing)


def find_local_pdfs_missing_in_xml(numbered, apr_by_date, code_to_name) -> list[str]:
    """List local numbered PDFs not referenced by any XML APr entry."""
    xml_pairs = set()
    for entries in apr_by_date.values():
        for e in entries:
            xml_pairs.add((code_to_name.get(e["code"], e["name"]), e["session"]))
    extra = []
    for (committee_name, session), pdf in numbered.items():
        if (committee_name, session) not in xml_pairs:
            extra.append(f"{committee_name}/{session}.pdf")
    return sorted(extra)


def find_update_orphans(update_pdfs, apr_by_date) -> list[str]:
    """List update/ PDFs whose (date) has no matching APr entry in XML."""
    orphans = []
    for (abbrev, d), pdf in update_pdfs.items():
        if d not in apr_by_date:
            orphans.append(f"{pdf.name} (no APr on {d.isoformat()})")
        else:
            codes_that_day = {e["code"] for e in apr_by_date[d]}
            if abbrev not in codes_that_day and abbrev.replace("SBü", "SBue") not in codes_that_day:
                orphans.append(f"{pdf.name} (APr exists on {d.isoformat()} but no code matches '{abbrev}')")
    return sorted(orphans)


def main() -> int:
    if not XML_PATH.exists():
        sys.exit(f"XML not found: {XML_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    apr_by_date, code_to_name, plenum_dates = parse_xml()
    numbered, update_pdfs, update_unparsed = index_local_protocols()
    audio_by_date, audio_stray, mp3_files, seq_meetings = index_audio()
    media = index_mediathek()

    # Build partial Mediathek room ↔ physical room mapping by cross-correlating
    # the 4 Sequence XMLs with Mediathek records on the same date.
    room_correlations: dict[str, list[dict]] = defaultdict(list)
    for sm in seq_meetings:
        if sm["start_dt"] is None:
            continue
        d = sm["start_dt"].date()
        same_day = [r for r in media if r["date"] == d]
        for r in same_day:
            r_dt = datetime.strptime(r["start_iso"], "%Y-%m-%dT%H:%M:%S")
            delta_min = (sm["start_dt"] - r_dt).total_seconds() / 60.0
            room_correlations[r["room"]].append(
                {
                    "mediathek_file": r["path"].name,
                    "physical_room": sm["room"],
                    "committee_code": sm["code"],
                    "session_title": sm["title"],
                    "mediathek_start": r["start_iso"],
                    "meeting_start": sm["start_dt"].isoformat(timespec="seconds"),
                    "delta_min": round(delta_min, 1),
                }
            )

    rows = build_rows(media, apr_by_date, audio_by_date, plenum_dates,
                      numbered, update_pdfs, code_to_name)

    # CSV
    fieldnames = [
        "mediathek_file", "room_id", "start_datetime",
        "candidate_committees", "candidate_protocol_pdfs",
        "candidate_audio_files", "notes",
    ]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Aggregate counts
    n_unique = sum(1 for r in rows if r["notes"].startswith("unique APr candidate"))
    n_multi = sum(1 for r in rows if " APr candidates" in r["notes"] and not r["notes"].startswith("unique"))
    n_none = sum(1 for r in rows if r["notes"].startswith("no APr") or r["notes"].startswith("plenum day (PlPr)"))
    n_plenum_collide = sum(1 for r in rows if "plenum day + APr" in r["notes"])
    rooms = sorted({r["room_id"] for r in rows})
    room_count = {rm: sum(1 for r in rows if r["room_id"] == rm) for rm in rooms}

    xml_pdfs_missing = find_xml_pdfs_missing_on_disk(apr_by_date, numbered, code_to_name)
    local_pdfs_extra = find_local_pdfs_missing_in_xml(numbered, apr_by_date, code_to_name)
    update_orphans = find_update_orphans(update_pdfs, apr_by_date)

    # Markdown report
    lines: list[str] = []
    a = lines.append
    a("# Mediathek → Ausschussprotokolle: mapping report")
    a("")
    a(f"Generated: {datetime.now().isoformat(timespec='seconds')}  ")
    a("Scope: 8. Wahlperiode (Landtag Brandenburg)")
    a("")
    a("## Data sources")
    a("")
    a(f"- **Mediathek**: {len(media)} `.mp4` recordings, room IDs {rooms}, dates "
      f"{min(r['date'] for r in media).isoformat()} → {max(r['date'] for r in media).isoformat()}")
    a(f"- **Ausschussprotokolle (numbered)**: {len(numbered)} local PDFs across "
      f"{len({k[0] for k in numbered})} committees")
    a(f"- **Ausschussprotokolle/update/**: {len(update_pdfs)} parsed PDFs"
      + (f", {len(update_unparsed)} unparsable" if update_unparsed else ""))
    a(f"- **Tondateien**: {sum(len(v) for v in audio_by_date.values())} `.nsv` files "
      f"across {len({a_['committee'] for v in audio_by_date.values() for a_ in v})} committees; "
      f"plus {len(mp3_files)} `.mp3` and {len(seq_meetings)} `.xml` Sequence companion files")
    a(f"- **exportWP8.xml**: {sum(len(v) for v in apr_by_date.values())} unique APr entries "
      f"on {len(apr_by_date)} distinct dates; {len(plenum_dates)} plenum dates")
    a("")
    a("## How the join works")
    a("")
    a("The Parlamentsspiegel export (`exportWP8.xml`) is the spine. Each "
      "`<Dokument>` with `<DokArt>APr</DokArt>` gives the canonical "
      "`(committee_code, session_number, date, Urheber)` tuple, where the "
      "committee code is parsed from `<LokURL>…/parladoku/w8/apr/<CODE>/<n>.pdf`. "
      "Local Ausschussprotokoll PDFs at `Ausschussprotokolle 8. Wahlperiode (…)/"
      "<committee>/<n>.pdf` match those tuples by `(Urheber, session_number)`. "
      "Tondateien filenames carry the date and the committee is the parent folder. "
      "Mediathek filenames `record_<room>_<channel>_start_<date>-<time>.mp4` only "
      "carry a numeric room ID; the room→committee mapping is **not present in "
      "the local data**.")
    a("")
    a("**Therefore the only deterministic key from local files is the date.** When "
      "several committees meet on the same date, every candidate is listed in the "
      "CSV; no automated choice is made.")
    a("")
    a("## Mediathek room IDs")
    a("")
    a(f"Three room IDs appear: {', '.join(rooms)}. Counts: " +
      ", ".join(f"{rm}={room_count[rm]}" for rm in rooms) + ".")
    a("The shared channel ID `84693` is constant. These IDs are streaming "
      "identifiers from the Landtag Brandenburg media server and are **not** "
      "documented in this repository (no JSON/YAML/script mentions them).")
    a("")
    a("### Partial room mapping from Sequence XMLs")
    a("")
    a(f"The Tondateien folders contain {len(seq_meetings)} `.xml` Sequence "
      "companion files that record the physical room (`<MeetingRoom>`) and "
      "exact `<MeetingStart>` / `<MeetingEnd>` of the session. Cross-"
      "correlating those with Mediathek recordings on the same date yields "
      "the following pairings (Mediathek `start` is typically 30–60 minutes "
      "before the official `MeetingStart`, consistent with a stream that "
      "begins recording before the gavel):")
    a("")
    if room_correlations:
        a("| Mediathek room | Physical room | Committee | Session | Mediathek start | Meeting start | Δ (min before meeting) |")
        a("|---|---|---|---|---|---|---|")
        for rm in sorted(room_correlations):
            for c in room_correlations[rm]:
                a(f"| `{rm}` | `{c['physical_room']}` | {c['committee_code']} | "
                  f"{c['session_title']} | {c['mediathek_start']} | "
                  f"{c['meeting_start']} | {c['delta_min']} |")
        a("")
        observed = {rm: sorted({c["physical_room"] for c in room_correlations[rm]})
                    for rm in room_correlations}
        a("**Observed (n samples is small — treat as hints, not a proven mapping):**")
        for rm, phys in sorted(observed.items()):
            a(f"- Mediathek `{rm}` → physical room(s) " + ", ".join(f"`{p}`" for p in phys))
        unmapped = [rm for rm in rooms if rm not in room_correlations]
        if unmapped:
            a("- " + ", ".join(f"`{rm}`" for rm in unmapped)
              + " — no Sequence XML available; physical room unknown.")
    else:
        a("*(no correlations could be formed)*")
    a("")
    a("A full Mediathek-room → physical-room mapping would need a Sequence "
      "XML for each committee×room combination (currently only 4 sessions are "
      "covered out of 70 Mediathek recordings) or an external lookup against "
      "the Landtag media-portal URL scheme.")
    a("")
    a("## Outcome summary")
    a("")
    a(f"- Mediathek rows produced: **{len(rows)}**")
    a(f"- Unique APr candidate (single committee met that day): **{n_unique}**")
    a(f"- Multiple APr candidates (ambiguous by date alone): **{n_multi}**")
    a(f"- No APr that date (incl. plenum-only days): **{n_none}**")
    a(f"- Plenum day with APr also on same date: **{n_plenum_collide}**")
    a("")
    a("Full per-file detail is in `tmp/mediathek_to_protocol.csv`.")
    a("")
    a("## Inconsistencies and gaps")
    a("")

    # 1. dates with no APr
    no_apr_dates = sorted({r["start_datetime"][:10] for r in rows
                           if r["notes"].startswith("no APr") or r["notes"].startswith("plenum day (PlPr)")})
    a("### Mediathek recordings on dates with no Ausschussprotokoll")
    a("")
    if no_apr_dates:
        for d_str in no_apr_dates:
            d_ = datetime.strptime(d_str, "%Y-%m-%d").date()
            flag = " (Plenum day)" if d_ in plenum_dates else ""
            files_on = [r for r in rows if r["start_datetime"].startswith(d_str)]
            a(f"- **{d_str}**{flag} — {len(files_on)} recording(s): "
              + ", ".join(f["mediathek_file"] for f in files_on))
    else:
        a("- *(none)*")
    a("")

    # 2. plenum collisions
    a("### Mediathek dates that are both Plenum and committee days")
    a("")
    plenum_collide_dates = sorted({r["start_datetime"][:10] for r in rows if "plenum day + APr" in r["notes"]})
    if plenum_collide_dates:
        for d_str in plenum_collide_dates:
            files_on = [r for r in rows if r["start_datetime"].startswith(d_str)]
            a(f"- **{d_str}** — Mediathek may belong to either Plenarprotokoll or "
              f"one of the APr that day; {len(files_on)} recording(s) on this date.")
    else:
        a("- *(none)*")
    a("")

    # 3. XML-referenced PDFs missing locally
    a("### APr entries in XML with no matching local PDF")
    a("")
    if xml_pdfs_missing:
        a(f"{len(xml_pdfs_missing)} entries — the canonical Parlamentsspiegel "
          "lists protocols that aren't present in the local read-only data drop "
          "(expected: local is a curated subset).")
        for line in xml_pdfs_missing[:25]:
            a(f"- {line}")
        if len(xml_pdfs_missing) > 25:
            a(f"- … and {len(xml_pdfs_missing) - 25} more (see XML for full list)")
    else:
        a("- *(none)*")
    a("")

    # 4. local PDFs not in XML
    a("### Local numbered PDFs not referenced by any XML APr entry")
    a("")
    if local_pdfs_extra:
        for line in local_pdfs_extra:
            a(f"- {line}")
    else:
        a("- *(none)*")
    a("")

    # 5. update/ orphans
    a("### `update/` PDFs without a matching XML APr entry")
    a("")
    if update_orphans:
        for line in update_orphans:
            a(f"- {line}")
    else:
        a("- *(none)*")
    a("")

    # 6. stray audio
    a("### Tondateien with non-standard filenames")
    a("")
    if audio_stray:
        for nsv in audio_stray:
            a(f"- {nsv.relative_to(REPO_ROOT)}")
    else:
        a("- *(none)*")
    a("")

    # 7. unparsed update
    a("### `update/` PDFs whose filename did not parse")
    a("")
    if update_unparsed:
        for pdf in update_unparsed:
            a(f"- {pdf.name}")
    else:
        a("- *(none)*")
    a("")

    a("## Reasoning")
    a("")
    a("The mapping reasoning is *inconsistent with the data alone* for two "
      "reasons:")
    a("")
    a("1. **Mediathek room IDs are not resolvable locally.** The repo holds no "
      "table tying `5737`, `11604`, `12510` to physical committee rooms, so the "
      "filename cannot identify the committee directly. Only the date (and "
      "optionally the start time, which the audio files lack) is shared with the "
      "protocol metadata.")
    a("")
    a("2. **Parliamentary days are densely scheduled.** On many dates 2–5 "
      "committees sit in parallel (e.g., 2025‑01‑15 has five APr entries). A "
      "single Mediathek file on that date matches every one of them by the only "
      "available key, so any non-ambiguous answer requires an extra signal "
      "(room → committee mapping, PDF start-time text extraction, or manual "
      "annotation).")
    a("")
    a("There are also outright **structural gaps**: Mediathek recordings exist "
      "on dates with no APr and no Tondatei (notably 2025‑02‑05 — three "
      "consecutive starts within 15 minutes, consistent with stream restarts "
      "rather than three sessions — and 2025‑05‑16). These likely correspond to "
      "non-committee broadcasts (press conferences, hearings, technical tests) "
      "that the Parlamentsspiegel does not index as APr.")
    a("")
    a("To close the loop, recommended follow-ups (out of scope here):")
    a("- Obtain the Landtag media server's catalogue (page that lists "
      "`record_<roomID>_…` URLs with a human-readable session label) and "
      "permanently encode the `roomID → committee` mapping.")
    a("- Extract the first-page text of each protocol PDF (`Beginn: HH:MM Uhr`) "
      "and compare to Mediathek `start_datetime` to disambiguate the parallel-"
      "committee days.")
    a("")

    MD_OUT.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"Wrote {CSV_OUT.relative_to(REPO_ROOT)} ({len(rows)} rows) and "
        f"{MD_OUT.relative_to(REPO_ROOT)}. "
        f"unique={n_unique} multi={n_multi} none={n_none} plenum_collide={n_plenum_collide}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
