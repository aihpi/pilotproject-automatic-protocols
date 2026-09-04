#!/usr/bin/env python3
"""Transcribe meeting audio to Markdown with Whisper, optionally diarised.

Inputs (one per ``--input-list`` line) may be either:

* an **audio file** (``.wav``/``.mp4``/…) — decoded to 16 kHz mono and turned
  into a log-mel spectrogram inline; or
* a precomputed **``.npy`` mel** from ``scripts/prepare_audio.py`` (shape
  [n_mels, T], 100 frames per second) — the original, GPU-light path.

Each is decoded with a chunked seek loop that mirrors
``whisper.transcribe.transcribe()`` but skips the audio→mel step. Segment
boundaries come from Whisper's own timestamp tokens; nothing fancier
(temperature fallback, no-speech threshold, hallucination guard) is wired in.

With ``--diarize`` (audio input only), pyannote speaker diarisation runs on the
same waveform and each Whisper segment is assigned to the speaker turn it most
overlaps; consecutive same-speaker segments are grouped under a ``<SD-SPK>``
header (the convention ``build_dataset.py`` already passes through). Without the
flag the output is byte-identical to the pre-diarisation transcriber.
"""

from __future__ import annotations

import argparse
import os
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


def load_input(path: Path, n_mels: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (mel tensor, waveform-or-None) for an audio file or a .npy mel.

    ``.npy`` inputs are loaded as-is (no waveform available, so the second
    element is None). Any other suffix is treated as audio: decoded to 16 kHz
    mono via ffmpeg and turned into a log-mel spectrogram inline. The waveform
    (shape ``(1, T)`` at ``whisper.audio.SAMPLE_RATE``) is returned so
    diarisation can run on the same audio without re-decoding.
    """
    if path.suffix.lower() == ".npy":
        return torch.from_numpy(np.load(path)), None
    audio = whisper.audio.load_audio(str(path))  # ffmpeg -> 16 kHz mono float32
    mel = whisper.audio.log_mel_spectrogram(audio, n_mels=n_mels)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T) for pyannote
    return mel, waveform


def run_diarization(waveform: torch.Tensor, sample_rate: int, pipeline, *,
                    num_speakers=None, min_speakers=None,
                    max_speakers=None) -> list[dict]:
    """Run pyannote diarisation; return turns ``[{start, end, speaker}]`` by start.

    The waveform is passed in-memory (``{"waveform", "sample_rate"}``) rather than
    as a file path, so diarisation does not depend on pyannote's file decoder.
    """
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
    # pyannote 4.x returns a DiarizeOutput (.speaker_diarization is the Annotation);
    # 3.x / legacy mode returns the Annotation directly.
    annotation = getattr(result, "speaker_diarization", result)
    turns = [
        {"start": float(seg.start), "end": float(seg.end), "speaker": str(label)}
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])
    return turns


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Tag each segment with the speaker turn it most overlaps (in place).

    Segments arrive in time order, so a single forward cursor over the sorted
    turns makes this O(n + m). Segments overlapping no turn (diarisation gaps)
    inherit the previous segment's speaker, or the nearest turn for the first.
    """
    if not turns:
        for seg in segments:
            seg["speaker"] = "UNKNOWN"
        return segments

    n = len(turns)
    cursor = 0
    prev_speaker: str | None = None
    for seg in segments:
        while cursor < n and turns[cursor]["end"] <= seg["start"]:
            cursor += 1
        best_speaker, best_ov = None, 0.0
        k = cursor
        while k < n and turns[k]["start"] < seg["end"]:
            ov = min(seg["end"], turns[k]["end"]) - max(seg["start"], turns[k]["start"])
            if ov > best_ov:
                best_ov, best_speaker = ov, turns[k]["speaker"]
            k += 1
        if best_speaker is None:
            if prev_speaker is not None:
                best_speaker = prev_speaker
            else:
                mid = 0.5 * (seg["start"] + seg["end"])
                best_speaker = min(
                    turns,
                    key=lambda t: abs(0.5 * (t["start"] + t["end"]) - mid),
                )["speaker"]
        seg["speaker"] = best_speaker
        prev_speaker = best_speaker
    return segments


def group_turns(segments: list[dict]) -> list[dict]:
    """Collapse consecutive same-speaker segments into one ``<SD-SPK>`` turn."""
    groups: list[dict] = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        if groups and groups[-1]["speaker"] == speaker:
            groups[-1]["end"] = seg["end"]
            groups[-1]["segments"].append(seg)
        else:
            groups.append({"speaker": speaker, "start": seg["start"],
                           "end": seg["end"], "segments": [seg]})
    return groups


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


def _segment_line(seg: dict) -> str | None:
    text = seg["text"].replace("\n", " ").strip()
    if not text:
        return None
    return f"[{format_ts(seg['start'])} --> {format_ts(seg['end'])}] {text}"


def write_md(
    out_path: Path,
    *,
    spec_path: Path,
    segments: list[dict],
    duration_s: float,
    model_id: str,
    language: str,
    groups: list[dict] | None = None,
    diarization_model: str | None = None,
) -> None:
    """Write the transcript Markdown.

    When ``groups`` is None the body is the legacy flat list of segment lines
    (byte-identical to the pre-diarisation output). When ``groups`` is given,
    each speaker turn is emitted under a ``<SD-SPK>`` header.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("---")
    lines.append(f"spectrogram: {spec_path}")
    lines.append(f"model: {model_id}")
    lines.append(f"language: {language}")
    lines.append(f"duration: {format_ts(duration_s)}")
    if groups is not None:
        lines.append(f"diarization: {diarization_model}")
        lines.append(f"num_speakers: {len({g['speaker'] for g in groups})}")
    lines.append(f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("---")
    lines.append("")
    if groups is None:
        for seg in segments:
            line = _segment_line(seg)
            if line is not None:
                lines.append(line)
    else:
        for group in groups:
            seg_lines = [ln for seg in group["segments"]
                         if (ln := _segment_line(seg)) is not None]
            if not seg_lines:
                continue
            lines.append(f"<SD-SPK>{group['speaker']}</SD>")
            lines.extend(seg_lines)
            lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-list", required=True, type=Path,
                   help="Manifest file: one audio (.wav/.mp4/…) or .npy mel path per line")
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
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split the manifest into this many shards (default: 1 = all). "
                        "Use with a SLURM job array to transcribe files in parallel, "
                        "one GPU per shard, across possibly-different nodes.")
    p.add_argument("--shard-index", type=int, default=0,
                   help="Which shard this process handles (0-based); takes manifest "
                        "entries [shard-index::num-shards].")
    p.add_argument("--diarize", action="store_true",
                   help="Run pyannote speaker diarisation (audio input only)")
    p.add_argument("--diarization-model",
                   default="pyannote/speaker-diarization-community-1",
                   help="pyannote pipeline id "
                        "(default: pyannote/speaker-diarization-community-1)")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="Exact number of speakers (optional)")
    p.add_argument("--min-speakers", type=int, default=None,
                   help="Lower bound on speaker count (optional)")
    p.add_argument("--max-speakers", type=int, default=None,
                   help="Upper bound on speaker count (optional)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token for the gated pyannote model "
                        "(default: $HF_TOKEN)")
    args = p.parse_args()

    inputs = read_manifest(args.input_list)
    if not inputs:
        print(f"manifest {args.input_list} is empty", file=sys.stderr)
        return 1

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        print(f"invalid sharding: shard-index={args.shard_index} "
              f"num-shards={args.num_shards}", file=sys.stderr)
        return 1
    if args.num_shards > 1:
        inputs = inputs[args.shard_index::args.num_shards]
        print(f"shard {args.shard_index}/{args.num_shards}: {len(inputs)} file(s)",
              file=sys.stderr)
        if not inputs:
            print("nothing to do for this shard", file=sys.stderr)
            return 0

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading whisper '{args.model}' on {device}", file=sys.stderr, flush=True)
    model = whisper.load_model(args.model, device=device)
    n_mels = model.dims.n_mels

    pipeline = None
    if args.diarize:
        if not args.hf_token:
            print("--diarize needs a HuggingFace token (set HF_TOKEN or pass "
                  "--hf-token); the pyannote model is gated", file=sys.stderr)
            return 1
        from pyannote.audio import Pipeline
        print(f"loading diariser '{args.diarization_model}' on {device}",
              file=sys.stderr, flush=True)
        try:  # pyannote.audio 4.x uses `token=`, 3.x uses `use_auth_token=`
            pipeline = Pipeline.from_pretrained(
                args.diarization_model, token=args.hf_token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                args.diarization_model, use_auth_token=args.hf_token)
        pipeline.to(torch.device(device))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for spec in tqdm(inputs, desc="transcribe", unit="file"):
        out = args.out_dir / f"{spec.stem}.md"
        if out.exists() and not args.overwrite:
            continue
        try:
            mel, waveform = load_input(spec, n_mels)
            segs, dur = transcribe_mel(model, mel, language=args.language)
            groups = None
            if args.diarize:
                if waveform is None:
                    raise ValueError(
                        "--diarize requires audio input (.wav/.mp4/…); got a "
                        ".npy mel with no waveform")
                turns = run_diarization(
                    waveform, SAMPLE_RATE, pipeline,
                    num_speakers=args.num_speakers,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers)
                assign_speakers(segs, turns)
                groups = group_turns(segs)
            write_md(
                out,
                spec_path=spec,
                segments=segs,
                duration_s=dur,
                model_id=f"openai/whisper-{args.model}",
                language=args.language,
                groups=groups,
                diarization_model=args.diarization_model if args.diarize else None,
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
