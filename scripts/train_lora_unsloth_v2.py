#!/usr/bin/env python3
"""Kaggle-faithful Unsloth (Q)LoRA fine-tune for gemma-4-31B-it.

Mirrors Daniel Hanchen's `gemma4-31b-unsloth` Kaggle notebook as closely as our
task allows, instead of the bespoke approach in ``scripts/train_lora_unsloth.py``:

  * LoRA targeting via Unsloth's multimodal-safe ``finetune_*_layers`` flags
    (language layers only) rather than an explicit ``target_modules`` list that
    could also match the vision tower.
  * ``get_chat_template(tokenizer, "gemma-4")`` (non-thinking — our protocol
    targets carry no ``<|channel>thought`` reasoning trace) + standard
    ``apply_chat_template`` so the system prompt is its own turn.
  * ``standardize_data_formats`` + render to a ``"text"`` column, stripping the
    leading ``<bos>`` (the collator re-adds exactly one).
  * Notebook defaults: r=8/alpha=8, bs1/ga4, 1 epoch, lr 2e-4, adamw_8bit,
    wd 0.001, linear schedule, warmup_steps 5.
  * ``train_on_responses_only`` (assistant-turn loss).

Our-task deviations (all exposed as CLI flags): the hardened system prompt baked
into the data, ``--max-seq-len 65536`` + ``SFTConfig.max_length`` (else TRL
truncates the long records), eval on a val split, tensorboard logging, and the
same best-eval_loss checkpoint + early stopping as ``scripts/train_lora.py``
(the notebook saves the last checkpoint).

Run in the Unsloth venv (see scripts/train_lora_unsloth.sbatch). IMPORTANT:
``import unsloth`` must precede transformers/trl so its patches apply.
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
    p.add_argument("--base-model", default="google/gemma-4-31B-it",
                   help="HF id or local path (notebook uses unsloth/gemma-4-31B-it; we keep "
                        "google/* to match the other v2 adapters' base). Unsloth accepts both.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Run folder. Default: auto-named results/YYYYMMDD-HHMMSS")
    p.add_argument("--max-seq-len", type=int, default=65536,
                   help="Max sequence length in tokens (default: 65536, 65k cap)")
    p.add_argument("--load-in-4bit", action="store_true", default=True,
                   help="QLoRA 4-bit (notebook default). Use --no-4bit for 16-bit (bf16) LoRA.")
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    # LoRA — notebook values
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--chat-template", default="gemma-4",
                   help="Unsloth chat template (default gemma-4, non-thinking; notebook uses "
                        "gemma-4-thinking — wrong for our non-reasoning targets)")
    # Optimisation — notebook values
    p.add_argument("--epochs", type=float, default=1.0,
                   help="Notebook full-run value (1). Bump for more passes (early stopping guards).")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="-1 = use --epochs; set small (e.g. 5) for a smoke test")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--lr-scheduler", default="linear")
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--early-stopping-patience", type=int, default=3,
                   help="Stop after N evals without eval_loss improvement; keep the best "
                        "checkpoint (needs a val set). 0 disables. Inert at epochs=1.")
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
    from unsloth.chat_templates import (get_chat_template, standardize_data_formats,
                                        train_on_responses_only)
    from alt_utils import resolve_out_dir, write_run_log, write_run_readme

    args.out_dir = resolve_out_dir(args.out_dir)

    print(f"loading {args.base_model} via Unsloth "
          f"(4bit={args.load_in_4bit}, max_seq_len={args.max_seq_len})", file=sys.stderr, flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        dtype=None,                       # auto
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )
    # Language-only LoRA via Unsloth's multimodal-safe flags (NOT an explicit module list).
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
    )
    # gemma-4 (non-thinking) chat template; same <|turn>user / <|turn>model markers.
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    data_files = {"train": str(args.train_jsonl)}
    if args.val_jsonl and args.val_jsonl.exists():
        data_files["validation"] = str(args.val_jsonl)
    dataset = load_dataset("json", data_files=data_files)
    has_val = "validation" in dataset
    print(f"train examples: {len(dataset['train'])}"
          + (f", val: {len(dataset['validation'])}" if has_val else ""), file=sys.stderr)
    # exact system prompt baked into the data (recorded in train_log/README)
    system_prompt = next((m["content"] for m in dataset["train"][0]["messages"]
                          if m["role"] == "system"), "")

    # Normalise role/content, then render each conversation to a single 'text' string.
    # removeprefix('<bos>'): the collator adds exactly one <bos> at tokenisation, so the
    # template's leading <bos> must be stripped to avoid a double-bos (notebook detail).
    # standardize_data_formats takes a Dataset (not a DatasetDict); apply per split. Our
    # data is already OpenAI role/content, so this is largely a no-op / column rename.
    for split in list(dataset.keys()):
        try:
            dataset[split] = standardize_data_formats(dataset[split])
        except Exception as e:  # already normalised → fall back to raw messages
            print(f"standardize_data_formats skipped for {split} ({e})", file=sys.stderr)

    def formatting_prompts_func(examples):
        convos = examples.get("conversations", examples.get("messages"))
        return {"text": [tokenizer.apply_chat_template(
            c, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
            for c in convos]}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    sft_config = SFTConfig(
        output_dir=str(args.out_dir),
        dataset_text_field="text",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=not args.load_in_4bit,
        max_length=args.max_seq_len,    # OURS: required, else TRL truncates the long records
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        # OURS: keep the BEST eval_loss checkpoint (not the last/most-overfit), like train_lora.py
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        greater_is_better=False if has_val else None,
        save_total_limit=2,
        optim="adamw_8bit",
        report_to="tensorboard",
        seed=args.seed,
        dataset_num_proc=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        args=sft_config,
    )

    # Assistant-turn loss: mask everything up to and including "<|turn>model\n".
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    # Early stopping (keep the best eval_loss checkpoint). Needs a val set; inert at epochs=1.
    if has_val and args.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience))

    result = trainer.train()
    best_eval = trainer.state.best_metric if has_val else None
    best_eval_s = f"{best_eval:.4f}" if isinstance(best_eval, (int, float)) else "-"
    print(f"final train loss: {result.training_loss:.4f} | best eval_loss: {best_eval_s}",
          file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out_dir))       # LoRA adapter (best, when loaded)
    tokenizer.save_pretrained(str(args.out_dir))
    bits = "4-bit QLoRA" if args.load_in_4bit else "bf16 LoRA"
    rows = [
        ("base model", args.base_model),
        ("recipe", "kaggle gemma4-31b-unsloth (finetune_*_layers, gemma-4 template)"),
        ("bits", bits),
        ("chat template", args.chat_template),
        ("max seq len", args.max_seq_len),
        ("LoRA r / alpha / dropout", f"{args.lora_r} / {args.lora_alpha} / {args.lora_dropout}"),
        ("batch / grad-accum", f"{args.batch_size} / {args.grad_accum}"),
        ("lr / scheduler / warmup", f"{args.lr} / {args.lr_scheduler} / {args.warmup_steps} steps"),
        ("weight decay", args.weight_decay),
        ("epochs / max steps", f"{args.epochs} / {args.max_steps}"),
        ("train jsonl", str(args.train_jsonl)),
        ("assistant-only loss", "True (train_on_responses_only)"),
        ("early-stopping patience", args.early_stopping_patience if has_val else "n/a (no val)"),
        ("final train loss", f"{result.training_loss:.4f}"),
        ("best eval_loss", best_eval_s),
    ]
    write_run_log(args.out_dir, "unsloth", rows, system_prompt=system_prompt)
    write_run_readme(args.out_dir, "unsloth", args.base_model,
                     f"{bits}, Kaggle gemma-4 recipe (r{args.lora_r}/a{args.lora_alpha}, "
                     f"gemma-4 template), max_seq_len {args.max_seq_len}.", rows,
                     system_prompt=system_prompt)
    print(f"saved adapter to {args.out_dir}", file=sys.stderr)
    _ = torch  # silence unused if the 4-bit branch skips bf16
    return 0


if __name__ == "__main__":
    sys.exit(main())
