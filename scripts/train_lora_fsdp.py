#!/usr/bin/env python3
"""Minimal sharded LoRA fine-tune (FSDP / DeepSpeed ZeRO-3) — alt to train_lora.py.

Approach #3 of branch ``fix/LoRA_alternative_implementations``: instead of
``device_map="auto"`` (which packs the embedding + lm_head onto GPU 0, leaving
no room before the logits even allocate), shard weights/grads/optimizer across
GPUs with **FSDP** (or DeepSpeed ZeRO-3) via ``accelerate launch``. This frees
GPU 0 and is the handoff's recommended weight/state-sharding line (§6 A).

This script is launcher-agnostic: ``accelerate launch`` supplies the FSDP/ZeRO
plugin from a config file (see configs/), so the script just loads the model
**without** a device_map and lets accelerate wrap it.

Note: sharding fixes the weight/state side, not the seq×vocab logits tensor. Run
a reduced --max-seq-len (8k–16k) first to prove the harness; combine with a
non-materialising loss for longer context.

Same chat-``messages`` JSONL + adapter output as the other implementations.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--val-jsonl", type=Path, default=None)
    p.add_argument("--base-model", default="google/gemma-4-E2B-it")
    p.add_argument("--out-dir", type=Path, default=Path("results/fsdp_lora"))
    p.add_argument("--max-seq-len", type=int, default=16384,
                   help="Start reduced (8k-16k): sharding doesn't fix the logits term")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules",
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--exclude-modules",
                   default=r".*(vision_tower|audio_tower|vision_model|visual).*")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1)
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

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    print(f"loading {args.base_model} (no device_map; accelerate/FSDP shards it), "
          f"max_seq_len={args.max_seq_len}", file=sys.stderr, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        # No device_map: accelerate's FSDP/ZeRO plugin places + shards the model.
    )
    model.config.use_cache = False
    model.enable_input_require_grads()  # needed for grad-checkpointing + LoRA

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    data_files = {"train": str(args.train_jsonl)}
    if args.val_jsonl and args.val_jsonl.exists():
        data_files["validation"] = str(args.val_jsonl)
    dataset = load_dataset("json", data_files=data_files)
    has_val = "validation" in dataset
    print(f"train examples: {len(dataset['train'])}", file=sys.stderr)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        exclude_modules=args.exclude_modules or None,
    )

    sft_config = SFTConfig(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,  # handoff §2: eval-batch-8 OOMs at long ctx
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_len,
        assistant_only_loss=True,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        report_to="tensorboard",
        seed=args.seed,
        dataset_num_proc=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    result = trainer.train()
    print(f"final train loss: {result.training_loss:.4f}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))
    print(f"saved adapter to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
