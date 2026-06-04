#!/usr/bin/env python3
"""Convert mp4 recordings to log-mel spectrograms for Whisper Large V3.

For each input mp4, ffmpeg is invoked (via whisper.audio.load_audio) to decode
to 16 kHz mono float32 PCM, then whisper.audio.log_mel_spectrogram produces a
128-bin log-mel spectrogram which is saved as .npy.

Reads paths from a manifest (one path per line; blank lines and #-comments
skipped). Runs over the manifest in a multiprocessing pool.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import whisper
from tqdm import tqdm

N_MELS = 128  # Whisper Large V3 spec


def read_manifest(path: Path) -> list[Path]:
    lines = path.read_text().splitlines()
    return [Path(s.strip()) for s in lines if s.strip() and not s.lstrip().startswith("#")]


def _process(args: tuple[str, str, bool]) -> tuple[str, str | None]:
    src_str, out_str, overwrite = args
    src = Path(src_str)
    out = Path(out_str)
    try:
        if out.exists() and not overwrite:
            return src_str, None
        audio = whisper.audio.load_audio(str(src))  # ffmpeg → 16 kHz mono float32
        mel = whisper.audio.log_mel_spectrogram(audio, n_mels=N_MELS)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, mel.cpu().numpy().astype(np.float32))
        return src_str, None
    except Exception:
        return src_str, traceback.format_exc()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-list", required=True, type=Path,
                   help="Manifest file: one mp4 path per line")
    p.add_argument("--out-dir", type=Path, default=Path("data/transcripts/spectrograms"),
                   help="Directory for output .npy files (default: data/transcripts/spectrograms)")
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
                   help="Parallel worker processes (default: $SLURM_CPUS_PER_TASK or 4)")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if output .npy already exists")
    args = p.parse_args()

    inputs = read_manifest(args.input_list)
    if not inputs:
        print(f"manifest {args.input_list} is empty", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(src), str(args.out_dir / f"{src.stem}.npy"), args.overwrite)
            for src in inputs]

    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_process, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="prepare_audio"):
            src, err = fut.result()
            if err is not None:
                failures.append((src, err))

    if failures:
        print(f"\n{len(failures)} file(s) failed:", file=sys.stderr)
        for src, err in failures:
            print(f"\n--- {src} ---\n{err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
