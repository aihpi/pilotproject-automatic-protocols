#!/usr/bin/env python3
"""Generate a smart summary (Protokoll) from a transcript with a fine-tuned LLM.

Loads a base model (optionally 4-bit) plus a LoRA adapter from any of the LoRA
trainers (canonical ``scripts/train_lora_unsloth.py`` or the PEFT trainer in
``alternative_frameworks/``) and turns each input transcript into a protocol-style
Markdown summary. With ``--granularity per-top`` (default, matching training) the
transcript is split on numbered ``<SD-TOP>`` markers, each agenda item is
summarised separately and the sections are concatenated under ``Zu TOP N``
headings; ``document`` summarises the whole transcript in one pass.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from build_dataset import split_transcript_by_top
from utils.prompt_io import build_user_message, load_summary_prompt, render_transcript_text
from utils.model_utils import context_window
from preprocess_protocol import split_front_matter


def iter_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in (".md", ".txt"))
    return [path]


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    source = args.merged_model or args.base_model
    quant_config = None
    if args.bits == 4 and not args.merged_model:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"loading {source}", file=sys.stderr, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tok_source = args.adapter or args.merged_model or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(str(tok_source))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.adapter and not args.merged_model:
        from peft import PeftModel
        print(f"attaching adapter {args.adapter}", file=sys.stderr, flush=True)
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system: str, user: str, args: argparse.Namespace) -> str:
    import torch

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    # return_dict=True so we get input_ids/attention_mask explicitly; the bare
    # return_tensors="pt" form yields a BatchEncoding (no .shape) on current
    # transformers and crashes with an empty AttributeError.
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    input_ids = enc["input_ids"].to(model.device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(model.device)

    if input_ids.shape[-1] > args.max_seq_len:
        print(f"  WARNING: prompt {input_ids.shape[-1]} > max-seq-len {args.max_seq_len}; "
              f"truncating input", file=sys.stderr)
        input_ids = input_ids[:, -args.max_seq_len:]
        attn = attn[:, -args.max_seq_len:] if attn is not None else None

    streamer = None
    if args.stream:
        from transformers import TextStreamer
        streamer = TextStreamer(tokenizer, skip_prompt=True)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            # Anti-degeneration controls (default off → unchanged behaviour). The
            # repetition/echo loops seen on long per-TOP inputs are curbed by
            # repetition_penalty>1 and no_repeat_ngram_size>0. See
            # tmp/HANDOFF_repetition_fix.md.
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
    return tokenizer.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()


def summarise(model, tokenizer, transcript: str, system: str, args: argparse.Namespace) -> str:
    if args.granularity == "document":
        return generate(model, tokenizer, system,
                        build_user_message("Gesamtes Protokoll", render_transcript_text(transcript)), args)

    tops = split_transcript_by_top(transcript)
    if not tops:
        print("  no numbered TOPs found; falling back to whole-document", file=sys.stderr)
        return generate(model, tokenizer, system,
                        build_user_message("Gesamtes Protokoll", render_transcript_text(transcript)), args)

    sections: list[str] = []
    for n in sorted(tops):
        # Same deployment-format input as training (prompt_io.build_user_message).
        user = build_user_message(f"TOP {n}", render_transcript_text(tops[n]))
        body = generate(model, tokenizer, system, user, args)
        sections.append(f"## Zu TOP {n}\n\n{body}")
    return "\n\n".join(sections)


def write_md(out_path: Path, *, source: Path, base_model: str, adapter: str | None, body: str
             ) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "---",
        f"source: {source.name}",
        "kind: protocol-summary",
        f"base_model: {base_model}",
        f"adapter: {adapter or 'none'}",
        f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "---",
        "",
    ]
    out_path.write_text("\n".join(header) + body + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="A transcript .md/.txt file or a directory of them")
    p.add_argument("--out-dir", type=Path, default=Path("results/summaries"),
                   help="Directory for output .md summaries (default: results/summaries)")
    p.add_argument("--base-model", default="google/gemma-4-E2B-it",
                   help="Base model id/path (default: google/gemma-4-E2B-it; must match the "
                        "base the adapter was trained on, e.g. google/gemma-4-31B-it)")
    p.add_argument("--adapter", type=Path, default=None,
                   help="LoRA adapter directory (default: none = base model only)")
    p.add_argument("--merged-model", type=Path, default=None,
                   help="Pre-merged model dir (skips base+adapter loading)")
    p.add_argument("--bits", type=int, choices=(4, 16), default=4,
                   help="4 = load base in NF4, 16 = bf16 (default: 4)")
    p.add_argument("--granularity", choices=("document", "per-top"), default="per-top",
                   help="Summarise per agenda item or whole document (default: per-top)")
    p.add_argument("--max-new-tokens", type=int, default=6144,
                   help="Max generated tokens per call (default: 6144 — above the longest "
                        "per-TOP training target, 4761, so a long section isn't clipped)")
    p.add_argument("--max-seq-len", type=int, default=65536,
                   help="Max prompt length before truncation. Default: 65536 (65k, matching "
                        "the training cap). Pass 0 for the model's full context window "
                        "(auto-detected, e.g. gemma-4-31B-it = 262144).")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Sampling temperature; 0 = greedy (default: 0.3)")
    p.add_argument("--top-p", type=float, default=0.9, help="Nucleus top-p (default: 0.9)")
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="Penalty on repeated tokens; 1.0 = off. Try a GENTLE ~1.15 to curb "
                        "echo/loops (1.3 over-suppresses into character-salad) (default: 1.0)")
    p.add_argument("--no-repeat-ngram-size", type=int, default=0,
                   help="Block repeating n-grams of this size; 0 = off (recommended). Small "
                        "values (e.g. 3) forbid legitimate German phrases → salad (default: 0)")
    p.add_argument("--min-new-tokens", type=int, default=0,
                   help="Minimum generated tokens before EOS allowed (default: 0)")
    p.add_argument("--system-prompt-file", type=Path, default=None,
                   help="Custom system prompt file (default: built-in German prompt)")
    p.add_argument("--stream", action="store_true", help="Stream tokens to stderr while generating")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate even if output .md already exists")
    args = p.parse_args()

    inputs = iter_inputs(args.input)
    if not inputs:
        print(f"no .md/.txt inputs found at {args.input}", file=sys.stderr)
        return 1

    if not args.max_seq_len:  # None or 0 -> fall back to the model's full context window
        args.max_seq_len = context_window(str(args.merged_model or args.base_model))
    print(f"max-seq-len: {args.max_seq_len} tokens", file=sys.stderr)

    system = (args.system_prompt_file.read_text(encoding="utf-8").strip()
              if args.system_prompt_file else load_summary_prompt())

    model, tokenizer = load_model(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for src in tqdm(inputs, desc="summarise", unit="file"):
        out = args.out_dir / f"{src.stem}.md"
        if out.exists() and not args.overwrite:
            continue
        try:
            _, transcript = split_front_matter(src.read_text(encoding="utf-8"))
            body = summarise(model, tokenizer, transcript.strip(), system, args)
            write_md(out, source=src, base_model=args.base_model,
                     adapter=str(args.adapter) if args.adapter else None, body=body)
        except Exception as exc:
            failures.append((str(src), repr(exc)))
            print(f"\nERROR on {src.name}: {exc!r}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} file(s) failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
