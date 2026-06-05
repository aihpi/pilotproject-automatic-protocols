#!/usr/bin/env python3
"""Download example plenary videos + protocols from public Landtag sources.

The video/protocol link is the **plenary sitting number** (see ``manifest.tsv``):

* Video: a Mediathek page embeds a 3Q SDN player as
  ``data-provider="threeqsdn"`` / ``data-dataid="<uuid>"``. The stream resolves
  via ``https://playout.3qsdn.com/<dataid>``, which ``yt-dlp``'s built-in
  ``3qsdn`` extractor handles. We grab the lowest-resolution mp4 on purpose —
  only the audio is used downstream and full plenary videos are multi-GB.
* Protocol: a born-digital PDF at
  ``parlamentsdokumentation.brandenburg.de/.../parladoku/w8/plpr/<N>.pdf``,
  downloaded with a plain HTTP GET.

Output stems are kept consistent (``Plenum_8-<N>_Transkript`` /
``Plenum_8-<N>_Protokoll``) so ``build_dataset.normalise_stem`` pairs them.
Skips existing outputs unless ``--overwrite``; exit codes 0/1/2.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

WAHLPERIODE = 8
PROTOCOL_URL = (
    "https://www.parlamentsdokumentation.brandenburg.de/starweb/LBB/ELVIS/"
    "parladoku/w{wp}/plpr/{n}.pdf"
)
PLAYOUT_URL = "https://playout.3qsdn.com/{dataid}"
DATAID_RE = re.compile(r'data-dataid="([0-9a-fA-F-]{36})"')
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) example-data-downloader"

# Lowest resolution first; only the audio matters downstream. Prefer a
# *progressive* https mp4 (3Q SDN's http-mp4-* renditions) over the HLS variant:
# HLS needs ffmpeg muxing, and some static ffmpeg builds segfault on it, whereas
# a progressive file is fetched by yt-dlp's native downloader with no ffmpeg.
YTDLP_FORMAT = "worst[protocol=https][ext=mp4]/worst[ext=mp4]/worst"


def video_stem(number: int) -> str:
    return f"Plenum_{WAHLPERIODE}-{number}_Transkript"


def protocol_stem(number: int) -> str:
    return f"Plenum_{WAHLPERIODE}-{number}_Protokoll"


def read_manifest(path: Path) -> list[tuple[int, str]]:
    """Parse ``number<TAB>mediathek_url`` rows (``#`` comments and blanks skipped)."""
    rows: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"bad manifest row (need number<TAB>url): {raw!r}")
        rows.append((int(parts[0].strip()), parts[1].strip()))
    return rows


def fetch_html(url: str, *, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_dataid(html: str) -> str:
    """Extract the 3Q SDN data-id from a Mediathek page; fail loudly otherwise."""
    m = DATAID_RE.search(html)
    if not m:
        snippet = html[:500].replace("\n", " ")
        raise ValueError(f"no data-dataid found in page; first 500 chars: {snippet!r}")
    return m.group(1)


def playout_url(dataid: str) -> str:
    return PLAYOUT_URL.format(dataid=dataid)


def download_video(page_url: str, out_path: Path, *, overwrite: bool) -> bool:
    """Resolve the 3Q stream from a Mediathek page and fetch it with yt-dlp.

    Returns True if a download ran, False if skipped (already present).
    """
    if out_path.exists() and not overwrite:
        return False
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp not found on PATH (add it via `uv sync`)")
    dataid = parse_dataid(fetch_html(page_url))
    url = playout_url(dataid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", "-f", YTDLP_FORMAT,
        "--no-playlist", "--no-part",
        "-o", str(out_path),
        url,
    ]
    if overwrite:
        cmd.insert(1, "--force-overwrites")
    subprocess.run(cmd, check=True)
    return True


def download_protocol(number: int, out_path: Path, *, overwrite: bool) -> bool:
    """Download the plenary protocol PDF for a sitting number. Returns True if fetched."""
    if out_path.exists() and not overwrite:
        return False
    url = PROTOCOL_URL.format(wp=WAHLPERIODE, n=number)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        ctype = resp.headers.get("Content-Type", "")
        if "pdf" not in ctype.lower():
            raise ValueError(f"{url} returned Content-Type {ctype!r}, expected PDF")
        data = resp.read()
    out_path.write_bytes(data)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=Path(__file__).with_name("manifest.tsv"),
                   help="TSV of `number<TAB>mediathek_url` rows")
    p.add_argument("--out-dir", type=Path, default=Path("data_example"),
                   help="Root output dir (default: data_example)")
    p.add_argument("--only", type=int, nargs="*",
                   help="Restrict to these sitting numbers")
    p.add_argument("--skip-video", action="store_true",
                   help="Download only protocols (no video fetch)")
    p.add_argument("--skip-protocol", action="store_true",
                   help="Download only videos (no protocol fetch)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download even if the output file already exists")
    args = p.parse_args()

    try:
        rows = read_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        print(f"cannot read manifest {args.manifest}: {exc}", file=sys.stderr)
        return 1
    if args.only:
        wanted = set(args.only)
        rows = [r for r in rows if r[0] in wanted]
    if not rows:
        print("no manifest rows selected", file=sys.stderr)
        return 1

    mp4_dir = args.out_dir / "transcripts" / "mp4"
    pdf_dir = args.out_dir / "protocols" / "pdf"

    failures: list[tuple[str, str]] = []
    for number, page_url in tqdm(rows, desc="download", unit="sitting"):
        if not args.skip_protocol:
            try:
                got = download_protocol(
                    number, pdf_dir / f"{protocol_stem(number)}.pdf",
                    overwrite=args.overwrite)
                tqdm.write(f"protocol {number}: {'downloaded' if got else 'exists'}")
            except Exception as exc:
                failures.append((f"protocol {number}", repr(exc)))
                tqdm.write(f"ERROR protocol {number}: {exc!r}")
        if not args.skip_video:
            try:
                got = download_video(
                    page_url, mp4_dir / f"{video_stem(number)}.mp4",
                    overwrite=args.overwrite)
                tqdm.write(f"video {number}: {'downloaded' if got else 'exists'}")
            except Exception as exc:
                failures.append((f"video {number}", repr(exc)))
                tqdm.write(f"ERROR video {number}: {exc!r}")

    if failures:
        print(f"\n{len(failures)} download(s) failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
