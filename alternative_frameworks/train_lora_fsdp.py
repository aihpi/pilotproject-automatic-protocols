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
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Run folder. Default: auto-named results/YYYYMMDD-HHMMSS")
    p.add_argument("--max-seq-len", type=int, default=65536,
                   help="65k cap (default). Sharding doesn't fix the seq×vocab logits "
                        "term, so --cce is required at this length (start 8k-16k to smoke).")
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
    p.add_argument("--early-stopping-patience", type=int, default=3,
                   help="Stop after N evals without eval_loss improvement; keep the best "
                        "checkpoint (needs a val set). 0 disables.")
    p.add_argument("--cce", action="store_true",
                   help="Use cut-cross-entropy: patch the gemma-4 wrapper forward so the "
                        "seq×vocab(262144) logits tensor is never materialised. FSDP shards the "
                        "weights but NOT this logits term, so --cce is required for cap 32768 / "
                        "uncapped. Needs the cut-cross-entropy package (in the base venv).")
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # ponytail: shared helpers stayed in scripts/
    from utils.alt_utils import resolve_out_dir, write_run_log, write_run_readme
    # Reuse the PEFT path's validated assistant-only-loss helpers so the FSDP adapter
    # is trained identically (mask system+user, train on the assistant turn only).
    from train_lora import enable_assistant_only_loss, drop_overlong_records

    args.out_dir = resolve_out_dir(args.out_dir)

    # cuDNN's fused multi-head-attention SDPA kernel intermittently fails in the
    # backward pass at long seq len under FSDP ("mha_graph.execute(...).is_good()
    # to be true, but got false") — the per-batch variable seq length builds a new
    # cuDNN graph each step and some shape trips the bug. Disable the cuDNN SDPA
    # backend so attention falls back to the flash / mem-efficient kernels.
    torch.backends.cuda.enable_cudnn_sdp(False)

    if args.cce:
        # Patch the wrapper forward BEFORE loading so the loss path uses cut-cross-entropy
        # (no full seq×vocab logits). Required for FSDP at cap 32768 / uncapped.
        from gemma4_cce_patch import patch_gemma4_conditional_generation_cce
        patch_gemma4_conditional_generation_cce()

    print(f"loading {args.base_model} (no device_map; accelerate/FSDP shards it), "
          f"max_seq_len={args.max_seq_len}, cce={args.cce}", file=sys.stderr, flush=True)
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
    print(f"train examples: {len(dataset['train'])}", file=sys.stderr)

    # Assistant-only loss (consistent with train_lora.py): record the exact system
    # prompt baked into the data and install the generation-tagged gemma-4 template so
    # SFTTrainer masks system+user and trains on the assistant turn only. The branch's
    # _GEMMA4_GEN_TEMPLATE renders byte-identical to the stock template; the helper
    # raises if the base model isn't gemma-4-shaped, so a prompt mismatch can never train.
    sample_messages = dataset["train"][0]["messages"]
    system_prompt = next((m["content"] for m in sample_messages
                          if m["role"] == "system"), "")
    enable_assistant_only_loss(tokenizer, sample_messages)
    # Drop records that would truncate past their assistant turn at this max_seq_len
    # (else assistant_only_loss + CCE divides by zero on an empty mask).
    dataset["train"] = drop_overlong_records(dataset["train"], tokenizer,
                                             args.max_seq_len, "train")
    if "validation" in dataset:
        dataset["validation"] = drop_overlong_records(dataset["validation"], tokenizer,
                                                       args.max_seq_len, "val")
    has_val = "validation" in dataset

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
        # Train on the assistant turn only (mask system+user), via the generation-tagged
        # template installed above — consistent with train_lora.py.
        assistant_only_loss=True,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        # keep the BEST eval_loss checkpoint (not the last), like train_lora.py
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        greater_is_better=False if has_val else None,
        save_total_limit=2,
        # CCE returns logits=None; eval must be loss-only or the metric path crashes.
        prediction_loss_only=args.cce,
        report_to="tensorboard",
        seed=args.seed,
        dataset_num_proc=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )

    class _LossOnlySFTTrainer(SFTTrainer):
        """compute_loss without touching outputs.logits (CCE returns logits=None, which
        TRL's default token-accuracy metric would crash on). Mirrors train_lora.py."""
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs.pop("_prediction_loss_only", None)
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer_cls = _LossOnlySFTTrainer if args.cce else SFTTrainer
    trainer = trainer_cls(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Early stopping (keep the best eval_loss checkpoint). Needs a val set.
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
    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))
    if trainer.is_world_process_zero():
        loss_kind = "cut-cross-entropy" if args.cce else "standard cross-entropy"
        write_run_log(args.out_dir, "fsdp", [
            ("base model", args.base_model),
            ("loss", loss_kind),
            ("max seq len", args.max_seq_len),
            ("LoRA r / alpha / dropout", f"{args.lora_r} / {args.lora_alpha} / {args.lora_dropout}"),
            ("batch / grad-accum", f"{args.batch_size} / {args.grad_accum}"),
            ("epochs / max steps", f"{args.epochs} / {args.max_steps}"),
            ("train jsonl", str(args.train_jsonl)),
            ("assistant-only loss", "True"),
            ("early-stopping patience", args.early_stopping_patience if has_val else "n/a (no val)"),
            ("final train loss", f"{result.training_loss:.4f}"),
            ("best eval_loss", best_eval_s),
        ], system_prompt=system_prompt)
        write_run_readme(args.out_dir, "fsdp", args.base_model,
                         f"bf16 LoRA, FSDP-sharded, {loss_kind}, max_seq_len {args.max_seq_len}.", [
                             ("precision", "bf16 LoRA"),
                             ("loss", loss_kind),
                             ("max seq len", args.max_seq_len),
                             ("dataset", str(args.train_jsonl)),
                             ("LoRA r / alpha / dropout", f"{args.lora_r} / {args.lora_alpha} / {args.lora_dropout}"),
                             ("epochs / effective batch", f"{args.epochs} / {args.batch_size * args.grad_accum}"),
                             ("learning rate", args.lr),
                             ("assistant-only loss", "True"),
                             ("final train loss", f"{result.training_loss:.4f}"),
                             ("best eval_loss", best_eval_s),
                         ], system_prompt=system_prompt)
    print(f"saved adapter to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
