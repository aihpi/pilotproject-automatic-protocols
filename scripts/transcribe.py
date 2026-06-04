#!/usr/bin/env python3
"""Transcribe Whisper-Large-V3 log-mel spectrograms (.npy) to Markdown.

Inputs are .npy files produced by ``scripts/prepare_audio.py`` (shape
[128, T], dtype float32, 100 mel frames per second of audio).

Each input is decoded with a chunked seek loop that mirrors
``whisper.transcribe.transcribe()`` but skips the audio→mel step. Segment
boundaries come from Whisper's own timestamp tokens; nothing fancier
(temperature fallback, no-speech threshold, hallucination guard) is wired
in for v1.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import whisper
from tqdm import tqdm
from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    SAMPLE_RATE,
    pad_or_trim,
)
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import get_tokenizer
from whisper.utils import exact_div


def read_manifest(path: Path) -> list[Path]:
    lines = path.read_text().splitlines()
    return [Path(s.strip()) for s in lines if s.strip() and not s.lstrip().startswith("#")]


def format_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def transcribe_mel(
    model: "whisper.Whisper",
    mel: torch.Tensor,
    language: str,
) -> tuple[list[dict], float]:
    """Run chunked decoding on a precomputed mel; return (segments, duration_s).

    Each segment is ``{"start": float, "end": float, "text": str}``. The seek
    loop and timestamp-token parsing follow whisper.transcribe.transcribe()
    (without temperature fallback / no-speech / hallucination handling).
    """
    device = model.device
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    mel = mel.to(device).to(dtype)

    tokenizer = get_tokenizer(
        multilingual=model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task="transcribe",
    )

    options = DecodingOptions(
        task="transcribe",
        language=language,
        without_timestamps=False,
        fp16=(device.type == "cuda"),
    )

    input_stride = exact_div(N_FRAMES, model.dims.n_audio_ctx)  # = 2
    time_precision = input_stride * HOP_LENGTH / SAMPLE_RATE     # = 0.02 s

    content_frames = mel.shape[-1]
    duration_s = content_frames * HOP_LENGTH / SAMPLE_RATE

    segments: list[dict] = []
    seek = 0
    with tqdm(total=content_frames, unit="frames", leave=False, desc="decode") as pbar:
        while seek < content_frames:
            time_offset = seek * HOP_LENGTH / SAMPLE_RATE
            segment_size = min(N_FRAMES, content_frames - seek)
            mel_segment = pad_or_trim(mel[:, seek : seek + segment_size], N_FRAMES)
            mel_segment = mel_segment.to(device).to(dtype)

            result: DecodingResult = model.decode(mel_segment, options)
            tokens = torch.tensor(result.tokens)

            timestamp_tokens = tokens.ge(tokenizer.timestamp_begin)
            single_timestamp_ending = (
                len(tokens) >= 2 and timestamp_tokens[-2:].tolist() == [False, True]
            )
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)

            if len(consecutive) > 0:
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))
                last_slice = 0
                for cur in slices:
                    sliced = tokens[last_slice:cur]
                    start_pos = sliced[0].item() - tokenizer.timestamp_begin
                    end_pos = sliced[-1].item() - tokenizer.timestamp_begin
                    text_tokens = [t.item() for t in sliced if t.item() < tokenizer.eot]
                    segments.append({
                        "start": time_offset + start_pos * time_precision,
                        "end":   time_offset + end_pos * time_precision,
                        "text":  tokenizer.decode(text_tokens).strip(),
                    })
                    last_slice = cur
                if single_timestamp_ending:
                    advance = segment_size
                else:
                    last_pos = tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                    advance = last_pos * input_stride
            else:
                # No timestamp pairs: emit one segment for the whole window.
                seg_dur = segment_size * HOP_LENGTH / SAMPLE_RATE
                ts = tokens[timestamp_tokens.nonzero().flatten()]
                if len(ts) > 0 and ts[-1].item() != tokenizer.timestamp_begin:
                    last_pos = ts[-1].item() - tokenizer.timestamp_begin
                    seg_dur = last_pos * time_precision
                text_tokens = [t.item() for t in tokens if t.item() < tokenizer.eot]
                segments.append({
                    "start": time_offset,
                    "end":   time_offset + seg_dur,
                    "text":  tokenizer.decode(text_tokens).strip(),
                })
                advance = segment_size

            # Guard against pathological zero-advance to avoid an infinite loop.
            advance = max(advance, 1)
            seek += advance
            pbar.update(min(advance, content_frames - pbar.n))

    return segments, duration_s


def write_md(
    out_path: Path,
    *,
    spec_path: Path,
    segments: list[dict],
    duration_s: float,
    model_id: str,
    language: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("---")
    lines.append(f"spectrogram: {spec_path}")
    lines.append(f"model: {model_id}")
    lines.append(f"language: {language}")
    lines.append(f"duration: {format_ts(duration_s)}")
    lines.append(f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("---")
    lines.append("")
    for seg in segments:
        text = seg["text"].replace("\n", " ").strip()
        if not text:
            continue
        lines.append(f"[{format_ts(seg['start'])} --> {format_ts(seg['end'])}] {text}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-list", required=True, type=Path,
                   help="Manifest file: one .npy spectrogram path per line")
    p.add_argument("--out-dir", type=Path, default=Path("data/transcripts/md"),
                   help="Directory for output .md files (default: data/transcripts/md)")
    p.add_argument("--model", default="large-v3",
                   help="Whisper model name (default: large-v3)")
    p.add_argument("--language", default="de",
                   help="Language code (default: de)")
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available, else cpu)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-transcribe even if output .md already exists")
    args = p.parse_args()

    inputs = read_manifest(args.input_list)
    if not inputs:
        print(f"manifest {args.input_list} is empty", file=sys.stderr)
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading whisper '{args.model}' on {device}", file=sys.stderr, flush=True)
    model = whisper.load_model(args.model, device=device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for spec in tqdm(inputs, desc="transcribe", unit="file"):
        out = args.out_dir / f"{spec.stem}.md"
        if out.exists() and not args.overwrite:
            continue
        try:
            mel = torch.from_numpy(np.load(spec))
            segs, dur = transcribe_mel(model, mel, language=args.language)
            write_md(
                out,
                spec_path=spec,
                segments=segs,
                duration_s=dur,
                model_id=f"openai/whisper-{args.model}",
                language=args.language,
            )
        except Exception as exc:
            failures.append((str(spec), repr(exc)))
            print(f"\nERROR on {spec}: {exc!r}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} file(s) failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
