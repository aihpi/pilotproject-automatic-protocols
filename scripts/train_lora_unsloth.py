#!/usr/bin/env python3
"""Minimal Unsloth (Q)LoRA fine-tune — alternative to scripts/train_lora.py.

Approach #1 of branch ``fix/LoRA_alternative_implementations``: attack the
gemma-4 long-context OOM with **Unsloth** on a *single* GPU. Unsloth ships its
own memory-efficient fused cross-entropy + flash-attention and patches the model
so the seq×vocab logits tensor (the OOM root cause, gemma-4 vocab = 262144) is
never fully materialised. This is the smallest possible setup; add complexity
(lower cap, offloaded checkpointing, an ``unsloth/*-bnb-4bit`` repo) only if it
fails.

Reads the same chat-``messages`` JSONL produced by ``scripts/build_dataset.py``
and writes an adapter + tokenizer for ``scripts/infer_summary.py``, so results
are directly comparable to the other implementations.

IMPORTANT: ``import unsloth`` must run before transformers/trl so its patches
apply — keep that import at the very top.
"""
from __future__ import annotations

import unsloth  # noqa: F401  (must be imported first to install patches)
from unsloth import FastModel

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--val-jsonl", type=Path, default=None)
    p.add_argument("--base-model", default="google/gemma-4-E2B-it",
                   help="HF id or local path (default: small model for smoke; scale to "
                        "google/gemma-4-31B-it). Unsloth also accepts unsloth/* 4-bit repos.")
    p.add_argument("--out-dir", type=Path, default=Path("results/unsloth_lora"))
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--load-in-4bit", action="store_true", default=True,
                   help="QLoRA 4-bit (default for single-GPU 31B). Use --no-4bit for bf16.")
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules",
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="-1 = use --epochs; set small (e.g. 20) for a smoke test")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.train_jsonl.exists():
        print(f"{args.train_jsonl} not found", file=sys.stderr)
        return 1

    import os
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    print(f"loading {args.base_model} via Unsloth "
          f"(4bit={args.load_in_4bit}, max_seq_len={args.max_seq_len})", file=sys.stderr, flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        use_gradient_checkpointing="unsloth",  # Unsloth's offloaded checkpointing (long ctx)
        random_state=args.seed,
    )

    data_files = {"train": str(args.train_jsonl)}
    if args.val_jsonl and args.val_jsonl.exists():
        data_files["validation"] = str(args.val_jsonl)
    dataset = load_dataset("json", data_files=data_files)
    has_val = "validation" in dataset
    print(f"train examples: {len(dataset['train'])}"
          + (f", val: {len(dataset['validation'])}" if has_val else ""), file=sys.stderr)

    sft_config = SFTConfig(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        # HF defaults eval batch to 8 -> OOMs at long ctx (handoff §2); pin to train batch.
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=not args.load_in_4bit,  # 4-bit compute is bf16 internally; let Unsloth pick
        max_length=args.max_seq_len,
        # loss on the assistant turn only (mask the system+transcript prompt)
        assistant_only_loss=True,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        optim="adamw_8bit",
        report_to="tensorboard",
        seed=args.seed,
        dataset_num_proc=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
    )

    result = trainer.train()
    print(f"final train loss: {result.training_loss:.4f}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out_dir))       # LoRA adapter
    tokenizer.save_pretrained(str(args.out_dir))
    print(f"saved adapter to {args.out_dir}", file=sys.stderr)
    _ = torch  # silence unused if branch skips bf16
    return 0


if __name__ == "__main__":
    sys.exit(main())
