#!/usr/bin/env python3
"""Extract 16 kHz mono WAV audio from the example mp4 videos with ffmpeg.

The WAV is what ``transcribe.py --diarize`` consumes (Whisper computes its mel
inline and pyannote runs on the same waveform). 16 kHz mono matches Whisper's
expected input and keeps files small. Skips existing outputs unless
``--overwrite``; exit codes 0/1/2.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def iter_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() == ".mp4")
    return [path]


def extract_wav(src: Path, out: Path, *, sample_rate: int = 16000,
                overwrite: bool = False) -> bool:
    """mp4 -> mono WAV at ``sample_rate``. Returns True if ffmpeg ran."""
    if out.exists() and not overwrite:
        return False
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path("data_example/transcripts/mp4"),
                   help="An .mp4 file or a directory of .mp4 files")
    p.add_argument("--out-dir", type=Path,
                   default=Path("data_example/transcripts/wav"),
                   help="Output directory for .wav files")
    p.add_argument("--sample-rate", type=int, default=16000,
                   help="Target sample rate in Hz (default: 16000)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even if the output .wav already exists")
    args = p.parse_args()

    inputs = iter_inputs(args.input)
    if not inputs:
        print(f"no .mp4 inputs found at {args.input}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for src in tqdm(inputs, desc="mp4->wav", unit="file"):
        out = args.out_dir / f"{src.stem}.wav"
        try:
            got = extract_wav(src, out, sample_rate=args.sample_rate,
                              overwrite=args.overwrite)
            tqdm.write(f"{src.name}: {'extracted' if got else 'exists'}")
        except Exception as exc:
            failures.append((str(src), repr(exc)))
            tqdm.write(f"ERROR on {src.name}: {exc!r}")

    if failures:
        print(f"\n{len(failures)} file(s) failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
