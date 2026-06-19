#!/usr/bin/env python3
"""Generate a protocol from a transcript with an Unsloth-trained LoRA adapter.

Companion to ``scripts/infer_summary.py`` for the Unsloth track: Unsloth adapters
target Unsloth's patched linear classes (``Gemma4ClippableLinear`` …), so stock
``transformers``/``peft`` can't load them — they must be loaded through Unsloth's
``FastModel``. This is the minimal Unsloth-native inference path (document
granularity: whole transcript in one pass), mirroring infer_summary's prompt so
outputs are comparable.

Run in the Unsloth venv:
  .venv-unsloth/bin/python scripts/infer_unsloth.py --input <transcript.md> \
      --adapter results/<run> --max-seq-len 32768 --out-dir results/<...>
"""
from __future__ import annotations

import unsloth  # noqa: F401  (import first to install patches)
from unsloth import FastModel

import argparse
import re
import sys
from pathlib import Path

# Inlined copy of build_dataset.DEFAULT_SYSTEM_PROMPT (the source of truth) so
# this script needs only the Unsloth venv (build_dataset pulls rapidfuzz/docling).
# Keep IN SYNC with scripts/build_dataset.py when the prompt changes.
DEFAULT_SYSTEM_PROMPT = """Du bist Protokollführer/in eines Ausschusses. Wandle das wörtliche Sitzungstranskript in ein formelles Ausschussprotokoll im amtlichen Stil um.

Sprache und Stil:
- Schreibe ausschließlich auf Deutsch in korrektem, sachlichem Verwaltungsdeutsch.
- Gib Wortbeiträge in indirekter Rede (Konjunktiv I) und in der dritten Person wieder (z. B. „Er betont, dass …“, „Sie verweist darauf, dass …“).
- Nenne Sprecher/innen mit Name und Rolle/Fraktion, z. B. „Kristy Augustin (CDU)“, „Steffen Freiberg (Minister für Bildung, Jugend und Sport)“.

Formatierung:
- Gliedere nach Tagesordnungspunkten mit genau EINER Überschrift „## Zu TOP N:“ je Punkt (Nummer aus den <SD-TOP>-Markierungen).
- Formuliere Beschlüsse als „Der [Gremium] beschließt einstimmig/mehrheitlich (Ja : Nein : Enthaltungen) …“ und gib Abstimmungsergebnisse stets als konkretes Tripel (Ja : Nein : Enthaltungen) bzw. als „einstimmig“/„mehrheitlich“ an — niemals als leeren Platzhalter.
- Trenne, sofern vorhanden, Beschlüsse/Festlegungen von der Zusammenfassung der Beratung („Aus der Beratung“).

Umgang mit dem Rohmaterial (Transkript):
- Das Transkript ist eine automatische Verschriftlichung (ASR) mit Sprecher-Diarisierung; es enthält technische Markierungen und Erkennungsfehler, die NICHT ins Protokoll gehören.
- Übernimm KEINE Zeitstempel (z. B. „[00:12:34]“ oder „(00:00:00.000 --> …)“) und erzeuge auch keine eigenen Zeit- oder „Sitzungsbeginn“-Angaben.
- Entnimm die TOP-Nummer den <SD-TOP>-Markierungen, übernimm die Markierungen und Tags selbst (z. B. „<SD-SPK>“, „<SD-TOP>“, „SPEAKER_03“) aber nicht in den Text.
- Ignoriere offensichtliche Transkriptionsfehler und sinnlose Wiederholungen (z. B. mehrfach hintereinander „Vielen Dank.“); wiederhole sie nicht und werte sie nicht als Inhalt.

Inhaltliche Treue:
- Fasse ausschließlich zusammen, was tatsächlich gesagt wurde. Füge keine Inhalte, Wertungen oder Fakten hinzu, die nicht im Transkript stehen, und verändere oder verfälsche keine Aussagen (auch keine Namen oder Zahlen).
- Im Zweifel knapper und näher am Wortlaut bleiben."""

_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_TOP_TAG_RE = re.compile(r"<SD-TOP>(.*?)</SD>", re.DOTALL)
_TOP_NUM_RE = re.compile(r"(?i)\b(?:TOP|Tagesordnungspunkt)\s*(\d+)")


def split_by_top(text: str) -> list[tuple[str, str]]:
    """Split a transcript into (label, segment) chunks at each <SD-TOP> marker
    (minimal inline version of build_dataset.split_transcript_by_top)."""
    marks = list(_TOP_TAG_RE.finditer(text))
    if not marks:
        return [("", text)]
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        num = _TOP_NUM_RE.search(m.group(1))
        out.append((f"TOP {num.group(1)}" if num else m.group(1).strip(),
                    text[m.start():end]))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Transcript .md")
    p.add_argument("--adapter", required=True, type=Path, help="Unsloth adapter dir")
    p.add_argument("--out-dir", type=Path, default=Path("results/summaries_unsloth"))
    p.add_argument("--granularity", choices=("per-top", "document"), default="per-top",
                   help="per-top (default, matches training; short inputs avoid OOM) or document")
    p.add_argument("--max-seq-len", type=int, default=65536,
                   help="Max prompt length before truncation (default: 65536, 65k)")
    p.add_argument("--max-new-tokens", type=int, default=6144,
                   help="Max generated tokens per call (default: 6144 — above the longest "
                        "per-TOP training target, 4761)")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="1.0 = off; a gentle ~1.15 curbs echo loops (1.3 over-suppressed → salad)")
    p.add_argument("--no-repeat-ngram-size", type=int, default=0,
                   help="0 = off (recommended); small n-gram blocks cause character-salad")
    p.add_argument("--min-new-tokens", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"{args.input} not found", file=sys.stderr)
        return 1

    text = args.input.read_text(encoding="utf-8")
    body = _FRONT_MATTER_RE.sub("", text).strip()

    print(f"loading Unsloth adapter {args.adapter}", file=sys.stderr, flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=str(args.adapter),       # adapter_config points at the base model
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    FastModel.for_inference(model)
    # Stop on gemma-4's turn terminator "<turn|>" (id 106) as well as the base
    # <eos>, else generation runs to max_new_tokens and the tail fills with
    # repetition (handoff: late/missing EOS). ("<end_of_turn>" is not a gemma-4 token.)
    _eot = tokenizer.convert_tokens_to_ids("<turn|>")
    eos_ids = [i for i in (tokenizer.eos_token_id, _eot)
               if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id]

    if args.granularity == "per-top":
        segments = split_by_top(body)
    else:
        segments = [("", body)]

    def generate(seg_text: str) -> str:
        convo = [{"role": "user", "content": [{"type": "text",
                  "text": (DEFAULT_SYSTEM_PROMPT + "\n\n" + seg_text).strip()}]}]
        # truncate the prompt to the context budget (leave room for the completion)
        inputs = tokenizer.apply_chat_template(
            convo, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            truncation=True, max_length=max(256, args.max_seq_len - args.max_new_tokens),
        ).to(model.device)
        out = model.generate(
            input_ids=inputs, max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature, top_p=args.top_p, do_sample=True,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_ids or None,
        )
        return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    sections = []
    for label, seg in segments:
        print(f"generating {label or 'document'}", file=sys.stderr, flush=True)
        gen = generate(seg)
        sections.append(f"## Zu {label}:\n\n{gen}" if label else gen)
    result = "\n\n".join(sections)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / (args.input.stem + ".protokoll.md")
    out_path.write_text(result, encoding="utf-8")
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
