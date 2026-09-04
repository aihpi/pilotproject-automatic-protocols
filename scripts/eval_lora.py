#!/usr/bin/env python3
"""Evaluation runner: summarise an examples directory with one LoRA adapter.

One process loads the model + adapter once, then writes a summary for every
(example x decode-preset) into the run directory. Each example is a folder with
``<stem>_Transkript.md`` (tagged transcript) and, optionally, ``<stem>_Protokoll.md``
(gold protocol, used only by eval_report.py). Pick the backend with ``--framework``
and run it under that framework's venv:

  # Unsloth adapters (the production trainer; must load through FastModel)
  .venv-unsloth/bin/python scripts/eval_lora.py --framework unsloth \
      --adapter results/20260902-31b_cap48k --adapter-id cap48k \
      --max-seq-len 49152 --examples-dir test --run-dir data/test/<timestamp>

  # Stock PEFT adapters (transformers + peft)
  uv run python scripts/eval_lora.py --framework peft \
      --adapter results/<run> --adapter-id <id> --base-model google/gemma-4-31B-it

Preprocessing (system+user chat template, 2-pass TOP split) is identical across
backends and matches the training input; only the adapter and the decode preset
vary. scripts/eval_lora.sbatch wraps this for SLURM.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from utils import eval_io


def _read_base_from_adapter(adapter: Path) -> str | None:
    cfg = adapter / "adapter_config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text())["base_model_name_or_path"]
        except Exception:
            return None
    return None


# --- PEFT / FSDP backend ------------------------------------------------------
def backend_peft(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    base = args.base_model or _read_base_from_adapter(args.adapter)
    quant = None
    if args.bits == 4:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    print(f"[peft] loading base {base} (bits={args.bits})", file=sys.stderr, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), quantization_config=quant, dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(str(args.adapter))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[peft] attaching adapter {args.adapter}", file=sys.stderr, flush=True)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    # EOS fix: gemma-4 ends a turn with "<turn|>" (id 106), not the base <eos>. Without
    # it generate() never stops at the section end and runs to max_new_tokens every time
    # (slow + bloated output). Stop on either. (Same fix as the unsloth backend.)
    _eot = tok.convert_tokens_to_ids("<turn|>")
    eos_ids = [i for i in (tok.eos_token_id, _eot)
               if isinstance(i, int) and i >= 0 and i != tok.unk_token_id]

    def generate_fn(system: str, user: str, k: dict) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(model.device)
        attn = enc.get("attention_mask")
        attn = attn.to(model.device) if attn is not None else None
        if ids.shape[-1] > args.max_seq_len:
            print(f"  truncating prompt {ids.shape[-1]} -> {args.max_seq_len}", file=sys.stderr)
            ids = ids[:, -args.max_seq_len:]
            attn = attn[:, -args.max_seq_len:] if attn is not None else None
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=attn,
                max_new_tokens=k["max_new_tokens"], min_new_tokens=k["min_new_tokens"],
                do_sample=k["temperature"] > 0,
                temperature=k["temperature"] if k["temperature"] > 0 else None,
                top_p=k["top_p"], repetition_penalty=k["repetition_penalty"],
                no_repeat_ngram_size=k["no_repeat_ngram_size"],
                pad_token_id=tok.pad_token_id, eos_token_id=eos_ids or None)
        return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()

    return generate_fn, str(base)


# --- Unsloth backend ----------------------------------------------------------
def backend_unsloth(args):
    import unsloth  # noqa: F401  (install patches first)
    from unsloth import FastModel

    print(f"[unsloth] loading adapter {args.adapter}", file=sys.stderr, flush=True)
    model, tok = FastModel.from_pretrained(model_name=str(args.adapter),
                                           max_seq_length=args.max_seq_len,
                                           load_in_4bit=True)
    FastModel.for_inference(model)
    base = _read_base_from_adapter(args.adapter) or "unsloth-adapter"
    # FastModel returns gemma-4's multimodal `Gemma4Processor`; token-id ops + decode
    # live on the wrapped `.tokenizer`, while apply_chat_template stays on the processor.
    tk = getattr(tok, "tokenizer", tok)
    # EOS fix: gemma-4 ends a turn with "<turn|>" (id 106), not "<end_of_turn>"
    # (which isn't even a single token here) nor only the base <eos>. Stop on either,
    # else generation runs to max_new_tokens and the tail fills with repetition.
    eot = tk.convert_tokens_to_ids("<turn|>")
    eos_ids = [i for i in (tk.eos_token_id, eot)
               if isinstance(i, int) and i >= 0 and i != tk.unk_token_id]

    def generate_fn(system: str, user: str, k: dict) -> str:
        # Unsloth's Gemma tokenizer uses the multimodal chat template, which
        # requires typed content parts (not a bare string) and does not take a
        # separate system role — so we cram system+user into one user turn (as
        # the original infer_unsloth.py did). PEFT keeps a proper system role.
        convo = [{"role": "user", "content": [{"type": "text",
                  "text": (system + "\n\n" + user).strip()}]}]
        # return_dict=True: the processor returns a BatchFeature, not a bare tensor.
        enc = tok.apply_chat_template(convo, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt", truncation=True,
                                      max_length=max(256, args.max_seq_len - k["max_new_tokens"]))
        enc = {kk: vv.to(model.device) for kk, vv in enc.items() if hasattr(vv, "to")}
        n_in = enc["input_ids"].shape[1]
        out = model.generate(
            **enc, max_new_tokens=k["max_new_tokens"], min_new_tokens=k["min_new_tokens"],
            do_sample=k["temperature"] > 0,
            temperature=k["temperature"] if k["temperature"] > 0 else None,
            top_p=k["top_p"], repetition_penalty=k["repetition_penalty"],
            no_repeat_ngram_size=k["no_repeat_ngram_size"],
            eos_token_id=eos_ids or None)
        return tk.decode(out[0][n_in:], skip_special_tokens=True).strip()

    return generate_fn, str(base)


BACKENDS = {"peft": backend_peft, "unsloth": backend_unsloth}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framework", required=True, choices=tuple(BACKENDS))
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--adapter-id", required=True,
                   help="Short stable id used in output filenames")
    p.add_argument("--base-model", default=None,
                   help="Base model id/path (PEFT). Default: read from adapter_config.json")
    p.add_argument("--bits", type=int, choices=(4, 16), default=4,
                   help="PEFT base precision: 4=NF4, 16=bf16 (default: 4)")
    p.add_argument("--granularity", choices=("per-top", "document"), default="per-top")
    p.add_argument("--max-seq-len", type=int, default=65536,
                   help="Prompt truncation budget (default: 65536, 65k); mirror the "
                        "adapter's training cap")
    p.add_argument("--examples-dir", type=Path, default=Path("test"),
                   help="Stable example inputs (one folder per example with "
                        "*_Transkript.md + *_Protokoll.md). Default: the tracked, clean "
                        "held-out set test/ (old contaminated set: data/test/examples)")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Output folder for this eval run (default: data/test/<UTC ts>). "
                        "The eval matrix passes a shared timestamp so all adapters land "
                        "in one run folder, mirroring results/<ts>/.")
    p.add_argument("--decodes", default="baseline,antirep",
                   help="Comma list of decode presets (default: baseline,antirep)")
    p.add_argument("--only", default=None,
                   help="Only run examples whose folder name contains this substring "
                        "(e.g. 'short_ARD_1' for smoke tests)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    decodes = [d.strip() for d in args.decodes.split(",") if d.strip()]
    for d in decodes:
        if d not in eval_io.DECODE_PRESETS:
            print(f"unknown decode preset {d!r}", file=sys.stderr)
            return 1

    run_dir = args.run_dir or (Path("data/test")
                               / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    generate_fn, base = BACKENDS[args.framework](args)
    written = eval_io.run_test_set(
        generate_fn, adapter_id=args.adapter_id, framework=args.framework,
        base_model=base, granularity=args.granularity,
        examples_dir=args.examples_dir, run_dir=run_dir,
        decodes=decodes, only=args.only, overwrite=args.overwrite)
    print(f"\nwrote {len(written)} summary file(s) under {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
