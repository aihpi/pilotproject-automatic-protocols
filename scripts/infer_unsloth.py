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

# Inlined from build_dataset.DEFAULT_SYSTEM_PROMPT (kept in sync) so this script
# needs only the Unsloth venv (build_dataset pulls rapidfuzz/docling).
DEFAULT_SYSTEM_PROMPT = (
    "Du bist Protokollführer/in eines Ausschusses. "
    "Wandle das wörtliche Sitzungstranskript in ein formelles "
    "Ausschussprotokoll im amtlichen Stil um.\n\n"
    "Sprache und Stil:\n"
    "- Schreibe ausschließlich auf Deutsch in korrektem, sachlichem Verwaltungsdeutsch.\n"
    "- Gib Wortbeiträge in indirekter Rede (Konjunktiv I) und in der dritten Person wieder.\n"
    "- Nenne Sprecher/innen mit Name und Rolle/Fraktion.\n\n"
    "Formatierung:\n"
    "- Gliedere nach Tagesordnungspunkten mit Überschriften „## Zu TOP N:“.\n"
    "- Formuliere Beschlüsse mit Abstimmungstripel (Ja : Nein : Enthaltungen).\n"
    "- Trenne Beschlüsse/Festlegungen von der Zusammenfassung der Beratung.\n\n"
    "Inhaltliche Treue:\n"
    "- Fasse ausschließlich zusammen, was tatsächlich gesagt wurde.\n"
    "- Im Zweifel knapper und näher am Wortlaut bleiben."
)

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
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
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
            temperature=args.temperature, top_p=args.top_p, do_sample=True,
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
