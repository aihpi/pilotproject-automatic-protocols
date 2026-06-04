#!/usr/bin/env python3
"""Fine-tune an instruction LLM with (Q)LoRA on transcript->protocol pairs.

Reads the JSONL chat datasets produced by ``scripts/build_dataset.py`` and trains
a LoRA adapter with TRL's ``SFTTrainer``. Defaults target Llama-3.3-70B-Instruct
in 4-bit (QLoRA) on a single 80 GB H100; the prompt (system + transcript) is
masked so loss is computed only on the protocol completion.

At 70B, 4-bit is effectively required on one GPU (16-bit LoRA would need ~140 GB);
``--bits 16`` is meant for smaller ``--base-model`` overrides. The adapter and
tokenizer are written to ``--out-dir`` for use by ``scripts/infer_summary.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True, type=Path,
                   help="Training JSONL (chat 'messages' records)")
    p.add_argument("--val-jsonl", type=Path, default=None,
                   help="Optional validation JSONL")
    p.add_argument("--base-model", default="google/gemma-4-E2B-it",
                   help="HF model id or local path (default: google/gemma-4-E2B-it for "
                        "testing; scale up with e.g. google/gemma-4-31B-it)")
    p.add_argument("--out-dir", type=Path, default=Path("results/lora_adapter"),
                   help="Directory for the trained adapter (default: results/lora_adapter)")
    p.add_argument("--bits", type=int, choices=(4, 16), default=4,
                   help="4 = QLoRA (NF4), 16 = bf16 LoRA (default: 4)")
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="Max packed sequence length in tokens (default: 4096)")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    p.add_argument("--target-modules",
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Comma-separated module-name suffixes to adapt")
    p.add_argument("--exclude-modules",
                   default=r".*(vision_tower|audio_tower|vision_model|visual).*",
                   help="Regex of module names to NOT adapt. Default drops the vision/audio "
                        "towers of multimodal models (e.g. Gemma) whose projections share the "
                        "q_proj/k_proj names but use unsupported wrapper layers; harmless no-op "
                        "for text-only models. Pass '' to disable.")
    p.add_argument("--epochs", type=float, default=3.0, help="Training epochs (default: 3)")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Cap training at N optimizer steps (default: -1 = use --epochs); "
                        "set a small value for a quick smoke test")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-device train batch size (default: 1)")
    p.add_argument("--grad-accum", type=int, default=16,
                   help="Gradient accumulation steps (default: 16)")
    p.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio (default: 0.03)")
    p.add_argument("--lr-scheduler", default="cosine", help="LR scheduler (default: cosine)")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (default: 0.0)")
    p.add_argument("--attn", choices=("flash_attention_2", "sdpa", "eager"), default="sdpa",
                   help="Attention implementation (default: sdpa, which needs no extra build; "
                        "use flash_attention_2 only if flash-attn is installed)")
    p.add_argument("--device-map", default="auto",
                   help="HF device_map (default: auto = shard across all visible GPUs; needed "
                        "for 16-bit LoRA on a large model that exceeds one GPU). Use 'cuda:0' "
                        "to force a single GPU.")
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="Disable gradient checkpointing (uses more VRAM)")
    p.add_argument("--packing", action="store_true",
                   help="Pack multiple short examples per sequence (good for per-TOP data)")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a checkpoint directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--overwrite", action="store_true",
                   help="Train even if --out-dir already contains an adapter")
    return p.parse_args()


def build_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
    from peft import prepare_model_for_kbit_training

    quant_config = None
    if args.bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"loading {args.base_model} (bits={args.bits}, attn={args.attn})",
          file=sys.stderr, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
        device_map=args.device_map,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ckpt = not args.no_gradient_checkpointing
    if args.bits == 4:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=ckpt)
    elif ckpt:
        # 16-bit LoRA + gradient checkpointing needs inputs to require grad, or
        # backward fails ("element 0 ... does not require grad"). kbit prep does
        # this for the 4-bit path; do it explicitly here.
        model.enable_input_require_grads()
    return model, tokenizer


def main() -> int:
    args = parse_args()

    if not args.train_jsonl.exists():
        print(f"{args.train_jsonl} not found", file=sys.stderr)
        return 1
    if (args.out_dir / "adapter_config.json").exists() and not args.overwrite:
        print(f"{args.out_dir} already has an adapter; use --overwrite", file=sys.stderr)
        return 1

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    data_files = {"train": str(args.train_jsonl)}
    if args.val_jsonl and args.val_jsonl.exists():
        data_files["validation"] = str(args.val_jsonl)
    dataset = load_dataset("json", data_files=data_files)
    print(f"train examples: {len(dataset['train'])}", file=sys.stderr)

    model, tokenizer = build_model_and_tokenizer(args)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        exclude_modules=args.exclude_modules or None,
    )

    has_val = "validation" in dataset
    sft_config = SFTConfig(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_length=args.max_seq_len,
        packing=args.packing,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        report_to="none",
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

    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    print(f"final train loss: {result.training_loss:.4f}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))
    print(f"saved adapter to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
