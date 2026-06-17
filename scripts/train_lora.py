#!/usr/bin/env python3
"""Fine-tune an instruction LLM with (Q)LoRA on transcript->protocol pairs.

Reads the JSONL chat datasets produced by ``scripts/build_dataset.py`` and trains
a LoRA adapter with TRL's ``SFTTrainer``; the prompt (system + transcript) is
masked so loss is computed only on the protocol completion.

Real runs use 16-bit LoRA (``--bits 16``, the default) which fits gemma-4-31B-it
on a single 80 GB H100; 4-bit QLoRA (``--bits 4``) is the smoke-test shortcut.
Each run is a self-contained, timestamp-named folder ``results/YYYYMMDD-HHMMSS/``
holding the adapter, tokenizer and a ``train_log.md`` of the run parameters (base
model and dataset live in the log content, not the folder name; smoke runs go to
``results/smoke_lora``); pass ``--out-dir`` to override. Training is logged to
TensorBoard under ``results/tensorboard``; with a validation set, the best model
is kept via early stopping. The adapter + tokenizer are written for
``scripts/infer_summary.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from model_utils import context_window


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True, type=Path,
                   help="Training JSONL (chat 'messages' records)")
    p.add_argument("--val-jsonl", type=Path, default=None,
                   help="Optional validation JSONL")
    p.add_argument("--base-model", default="google/gemma-4-E2B-it",
                   help="HF model id or local path (default: google/gemma-4-E2B-it for "
                        "testing; scale up with e.g. google/gemma-4-31B-it)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Adapter output dir. Default: auto-named results/YYYYMMDD_XX_lora for "
                        "16-bit runs (+ a YYYYMMDD_XX_log.md), results/smoke_lora for 4-bit "
                        "smoke runs. Pass a path to override.")
    p.add_argument("--bits", type=int, choices=(4, 16), default=16,
                   help="16 = bf16 LoRA (default, real runs), 4 = QLoRA NF4 (smoke test)")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="Max packed sequence length in tokens. Default: the base model's "
                        "context window (auto-detected, e.g. gemma-4-31B-it = 262144).")
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
    p.add_argument("--use-liger", action="store_true",
                   help="Enable Liger fused linear cross-entropy: computes the loss without "
                        "materialising the full seq×vocab logits tensor, so long --max-seq-len "
                        "fits on large-vocab models (gemma-4, vocab ~262k). Requires liger-kernel "
                        "and a supported model_type (gemma4_text is supported). NOTE: liger's "
                        "fused CE produces NaN grads on gemma-4 at long seq / bf16; prefer --cce.")
    p.add_argument("--cce", action="store_true",
                   help="Use cut-cross-entropy for the loss instead of liger (mutually exclusive "
                        "with --use-liger). Same memory benefit (no full logits) but numerically "
                        "stable at long seq / bf16; supports Gemma's logit softcap. Requires "
                        "cut-cross-entropy. The numerically-robust choice for gemma-4 long-context.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a checkpoint directory")
    p.add_argument("--early-stopping-patience", type=int, default=2,
                   help="Stop after N evals without eval_loss improvement and keep the best "
                        "model (default: 2; only active with a validation set)")
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

    if args.cce:
        # Cut-cross-entropy: patch the wrapper forward to compute the loss without
        # materialising full logits (numerically stable alternative to liger). No cheap
        # kernels are installed — only the loss path changes. See gemma4_cce_patch.py.
        from gemma4_cce_patch import patch_gemma4_conditional_generation_cce
        patch_gemma4_conditional_generation_cce()
    elif args.use_liger:
        # Patch the model CLASS *before* loading so the instance uses Liger's
        # fused-linear-CE forward: it fuses the lm_head matmul with cross-entropy in
        # chunks and never materialises the full seq×vocab logits tensor (the OOM at
        # long --max-seq-len on large-vocab models like gemma-4). The transformers
        # `use_liger_kernel` flag only swaps the cheap kernels on the *instance* and
        # leaves the standard full-logits loss path, so we apply it ourselves here.
        from transformers import AutoConfig
        from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
        cfg = AutoConfig.from_pretrained(args.base_model)
        text_cfg = getattr(cfg, "text_config", None)
        mt = getattr(text_cfg, "model_type", None) or cfg.model_type
        apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(mt)
        if apply_fn is None:
            print(f"liger: no kernel for model_type={mt}; using standard loss "
                  "(long sequences may OOM)", file=sys.stderr)
        else:
            # Installs liger's cheap kernels (RMSNorm/GEGLU/rope) on the text modules
            # AND its fused-CE forward on the *text* class (e.g. Gemma4ForCausalLM).
            apply_fn(fused_linear_cross_entropy=True, cross_entropy=False)
            print(f"liger: fused-linear-CE patch applied for model_type={mt} "
                  "(lm_head+CE fused; full logits not materialised)", file=sys.stderr)
        # gemma-4 loads as the multimodal wrapper Gemma4ForConditionalGeneration, which
        # liger does NOT patch (it only patches the text-only Gemma4ForCausalLM) — its
        # stock forward still materialises full logits. Patch the wrapper too so the
        # fused path actually engages. See scripts/gemma4_liger_patch.py.
        if cfg.model_type == "gemma4" and text_cfg is not None:
            from gemma4_liger_patch import patch_gemma4_conditional_generation
            patch_gemma4_conditional_generation()

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


def resolve_output(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """Decide the adapter dir and the log path written *inside* it.

    A run is a self-contained, timestamp-named folder ``results/YYYYMMDD-HHMMSS/``
    (no model/variant in the name — those live in the log content) holding the
    adapter, tokenizer and ``train_log.md``. explicit --out-dir -> use it (log
    still goes inside as ``train_log.md``); 4-bit smoke -> results/smoke_lora.
    """
    results = Path("results")
    if args.out_dir is not None:
        out = args.out_dir
        return out, out / "train_log.md"
    if args.bits == 4:
        out = results / "smoke_lora"
        return out, out / "train_log.md"
    out = results / datetime.now().strftime("%Y%m%d-%H%M%S")
    return out, out / "train_log.md"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_train_log(log_path: Path, args: argparse.Namespace, *, out_dir: Path,
                    logging_dir: str, n_train: int, n_val: int, train_loss: float,
                    best_metric, best_ckpt) -> None:
    """Write a Markdown log of the run parameters for later comparison."""
    eff_batch = args.batch_size * args.grad_accum
    rows = [
        ("date", datetime.now().isoformat(timespec="seconds")),
        ("git commit", _git_commit()),
        ("SLURM job", os.environ.get("SLURM_JOB_ID", "-")),
        ("base model", args.base_model),
        ("bits", f"{args.bits} ({'QLoRA NF4' if args.bits == 4 else 'bf16 LoRA'})"),
        ("adapter dir", str(out_dir)),
        ("tensorboard", logging_dir),
        ("LoRA r / alpha / dropout", f"{args.lora_r} / {args.lora_alpha} / {args.lora_dropout}"),
        ("target modules", args.target_modules),
        ("exclude modules", args.exclude_modules or "-"),
        ("learning rate", args.lr),
        ("lr scheduler / warmup", f"{args.lr_scheduler} / {args.warmup_ratio}"),
        ("weight decay", args.weight_decay),
        ("epochs / max steps", f"{args.epochs} / {args.max_steps}"),
        ("batch / grad-accum / effective", f"{args.batch_size} / {args.grad_accum} / {eff_batch}"),
        ("max seq len", args.max_seq_len),
        ("packing", args.packing),
        ("attention", args.attn),
        ("optimizer", "paged_adamw_8bit"),
        ("seed", args.seed),
        ("early-stopping patience", args.early_stopping_patience if n_val else "n/a (no val set)"),
        ("train jsonl", str(args.train_jsonl)),
        ("val jsonl", str(args.val_jsonl) if args.val_jsonl else "-"),
        ("records (train / val)", f"{n_train} / {n_val}"),
        ("final train loss", f"{train_loss:.4f}"),
        ("best eval_loss", f"{best_metric:.4f}" if best_metric is not None else "-"),
        ("best checkpoint", best_ckpt or "-"),
    ]
    lines = [f"# Training run — {out_dir.name}", "",
             "| Parameter | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote run log to {log_path}", file=sys.stderr)


def main() -> int:
    args = parse_args()

    if not args.train_jsonl.exists():
        print(f"{args.train_jsonl} not found", file=sys.stderr)
        return 1

    if args.max_seq_len is None:
        args.max_seq_len = context_window(args.base_model)
    out_dir, log_path = resolve_output(args)
    args.out_dir = out_dir
    # Collect all runs under one parent so `tensorboard --logdir results/tensorboard`
    # compares them. The TrainingArguments `logging_dir` field is deprecated and
    # ignored by the TB callback; the env var is the supported way to set it.
    logging_dir = str(Path("results/tensorboard") / out_dir.name)
    os.environ["TENSORBOARD_LOGGING_DIR"] = logging_dir
    print(f"adapter -> {out_dir} (bits={args.bits}, max-seq-len={args.max_seq_len}); "
          f"tensorboard -> {logging_dir}", file=sys.stderr)

    if (out_dir / "adapter_config.json").exists() and not args.overwrite:
        print(f"{out_dir} already has an adapter; use --overwrite", file=sys.stderr)
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
        # Match eval batch to train (default is 8): at long --max-seq-len, 8 sequences in
        # one eval forward OOMs GPU 0 even though batch-1 training fits. (Eval has no
        # gradient accumulation, so this just sets the eval forward batch.)
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        # Tell TRL the model uses a fused-CE forward that may return logits=None (true for
        # both --use-liger and --cce): TRL then skips its logits-based metric path in
        # compute_loss (which would crash on None logits) and reads token_accuracy from the
        # output instead. The actual fused forward is installed in build_model_and_tokenizer;
        # transformers' own liger application is a no-op for the gemma4 multimodal type, so
        # this flag only flips TRL's loss path (CCE returns no token_accuracy -> TRL warns).
        use_liger_kernel=args.use_liger or args.cce,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_length=args.max_seq_len,
        packing=args.packing,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        # Only eval_loss is used (early stopping + best model); no compute_metrics. With
        # the liger fused-CE eval path the model returns logits=None, so the trainer must
        # not gather eval logits — and skipping them avoids a full seq×vocab eval OOM.
        prediction_loss_only=has_val,
        # keep the best (lowest eval_loss) model when a val set is present
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        greater_is_better=False if has_val else None,
        save_total_limit=2,
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
    if has_val:
        from transformers import EarlyStoppingCallback
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    print(f"final train loss: {result.training_loss:.4f}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"saved adapter to {out_dir}", file=sys.stderr)

    if log_path is not None:
        write_train_log(
            log_path, args, out_dir=out_dir, logging_dir=logging_dir,
            n_train=len(dataset["train"]),
            n_val=len(dataset["validation"]) if has_val else 0,
            train_loss=result.training_loss,
            best_metric=trainer.state.best_metric,
            best_ckpt=trainer.state.best_model_checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
