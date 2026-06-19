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
                   help="Adapter output dir. Default: auto-named results/YYYYMMDD-HHMMSS for "
                        "real runs (with train_log.md + README.md inside), results/smoke_lora "
                        "for 4-bit smoke runs. Pass a path to override.")
    p.add_argument("--bits", type=int, choices=(4, 16), default=16,
                   help="16 = bf16 LoRA (default, real runs), 4 = QLoRA NF4 (smoke test)")
    p.add_argument("--max-seq-len", type=int, default=65536,
                   help="Max sequence length in tokens. Default: 65536 (65k cap, matching "
                        "build_dataset's default). Pass 0 to use the base model's full "
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
    p.add_argument("--cce", action="store_true",
                   help="Use cut-cross-entropy for the loss: computes it without materialising the "
                        "full seq×vocab logits tensor, so long --max-seq-len fits on large-vocab "
                        "models (gemma-4, vocab ~262k). Numerically stable at long seq / bf16 and "
                        "supports Gemma's logit softcap. Required for long-context gemma-4 training; "
                        "needs the cut-cross-entropy package. See scripts/gemma4_cce_patch.py.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a checkpoint directory")
    p.add_argument("--early-stopping-patience", type=int, default=3,
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
        # Cut-cross-entropy: patch the gemma-4 multimodal wrapper's forward to compute the
        # loss without materialising the full seq×vocab logits tensor (the long-seq OOM on
        # large-vocab models). Patch the CLASS *before* loading. See gemma4_cce_patch.py.
        from gemma4_cce_patch import patch_gemma4_conditional_generation_cce
        patch_gemma4_conditional_generation_cce()

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


# gemma-4 uses `<|turn>role … <turn|>` markers and renders the system message as
# its own turn. Its stock chat template has no `{% generation %}` block, which
# SFTConfig(assistant_only_loss=True) needs to know which tokens are the response.
# This template renders BYTE-IDENTICAL text to the stock one (asserted at runtime)
# but wraps the model turn in {% generation %} so loss falls on the assistant only.
_GEMMA4_GEN_TEMPLATE = (
    "{{ bos_token }}"
    "{%- for message in messages -%}"
    "{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}"
    "{{- '<|turn>' + role + '\n' -}}"
    "{%- if role == 'model' -%}"
    # literal "{% generation %}" markers (no whitespace-control dashes): TRL's
    # get_training_chat_template() requires the exact substring to accept the template.
    "{% generation %}{{- message['content'] | trim -}}{{- '<turn|>\n' -}}{% endgeneration %}"
    "{%- else -%}{{- message['content'] | trim -}}{{- '<turn|>\n' -}}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|turn>model\n' -}}{%- endif -%}"
)


def enable_assistant_only_loss(tokenizer, sample_messages: list[dict]) -> None:
    """Install a generation-tagged chat template so assistant_only_loss masks the
    prompt and trains on the assistant turn only. Fails fast (raises) if the new
    template doesn't render identically to the model's own — i.e. if the base model
    isn't gemma-4-shaped — so a run can never silently train on a mismatched prompt
    or an empty loss mask."""
    original = tokenizer.chat_template
    before = tokenizer.apply_chat_template(sample_messages, tokenize=False,
                                           add_generation_prompt=False)
    tokenizer.chat_template = _GEMMA4_GEN_TEMPLATE
    after = tokenizer.apply_chat_template(sample_messages, tokenize=False,
                                          add_generation_prompt=False)
    if before != after:
        tokenizer.chat_template = original
        raise RuntimeError(
            "assistant_only_loss: generation-tagged template does not match the "
            "model's chat template (base model may not be gemma-4). Aborting to "
            "avoid a train/inference prompt mismatch.")
    out = tokenizer.apply_chat_template(sample_messages, tokenize=True,
                                        return_assistant_tokens_mask=True, return_dict=True)
    n = sum(out.get("assistant_masks") or [])
    if n == 0:
        tokenizer.chat_template = original
        raise RuntimeError("assistant_only_loss: assistant mask is empty; aborting.")
    print(f"assistant-only loss: generation-tagged template installed "
          f"(render-identical; mask={n} tokens on the sample)", file=sys.stderr)


def drop_overlong_records(split, tokenizer, max_len: int, name: str):
    """Drop records whose full conversation tokenizes beyond ``max_len``.

    With assistant_only_loss, truncating such a record can remove its entire
    assistant turn, leaving zero unmasked tokens — which makes cut-cross-entropy's
    backward divide by zero (``grad_scale = 1 / lse.numel()``). Dropping them (with
    a logged count) is safe: a record whose target is truncated away can't train
    the response anyway. Records that fit are unaffected."""
    def fits(ex):
        ids = tokenizer.apply_chat_template(ex["messages"], tokenize=True,
                                            add_generation_prompt=False)
        return len(ids) <= max_len
    n0 = len(split)
    split = split.filter(fits)
    dropped = n0 - len(split)
    if dropped:
        print(f"{name}: dropped {dropped}/{n0} records longer than max_seq_len={max_len} "
              f"(would truncate the assistant away → CCE zero-token crash)", file=sys.stderr)
    if len(split) == 0:
        raise RuntimeError(f"{name}: all records exceed max_seq_len={max_len}; nothing to train on")
    return split


def _prompt_sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _prompt_section(system_prompt: str) -> str:
    """Markdown block recording the exact system prompt (for train_log / README)."""
    if not system_prompt:
        return "\n## System prompt\n\n_(none found in training data)_\n"
    return (f"\n## System prompt\n\nsha256: `{_prompt_sha(system_prompt)}`\n\n"
            f"```\n{system_prompt}\n```\n")


def write_train_log(log_path: Path, args: argparse.Namespace, *, out_dir: Path,
                    logging_dir: str, n_train: int, n_val: int, train_loss: float,
                    best_metric, best_ckpt, system_prompt: str = "") -> None:
    """Write a Markdown log of the run parameters for later comparison."""
    eff_batch = args.batch_size * args.grad_accum
    rows = [
        ("date", datetime.now().isoformat(timespec="seconds")),
        ("git commit", _git_commit()),
        ("SLURM job", os.environ.get("SLURM_JOB_ID", "-")),
        ("base model", args.base_model),
        ("bits", f"{args.bits} ({'QLoRA NF4' if args.bits == 4 else 'bf16 LoRA'})"),
        ("loss", "cut-cross-entropy" if args.cce else "standard cross-entropy"),
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
        ("assistant-only loss", "True"),
        ("system prompt (sha256)", _prompt_sha(system_prompt) if system_prompt else "-"),
    ]
    lines = [f"# Training run — {out_dir.name}", "",
             "| Parameter | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines.append(_prompt_section(system_prompt))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote run log to {log_path}", file=sys.stderr)


def write_run_readme(out_dir: Path, args: argparse.Namespace, *, n_train: int, n_val: int,
                     train_loss: float, best_metric, system_prompt: str = "") -> None:
    """Write a human-readable ``README.md`` of the run's key settings into the adapter dir
    (overwrites PEFT's generic model-card README). Companion to the full ``train_log.md``."""
    bits = f"{args.bits}-bit ({'QLoRA NF4' if args.bits == 4 else 'bf16 LoRA'})"
    loss = "cut-cross-entropy (--cce)" if args.cce else "standard cross-entropy"
    eff = args.batch_size * args.grad_accum
    best = f"{best_metric:.4f}" if best_metric is not None else "-"
    lines = [
        f"# LoRA adapter — {out_dir.name}",
        "",
        f"Fine-tuned `{args.base_model}` with {bits} + {loss}.",
        "",
        "| setting | value |",
        "|---|---|",
        f"| base model | `{args.base_model}` |",
        f"| dataset (train / val) | `{args.train_jsonl}` ({n_train} / {n_val}) |",
        f"| precision | {bits} |",
        f"| loss | {loss} |",
        f"| max seq len | {args.max_seq_len} |",
        f"| LoRA r / alpha / dropout | {args.lora_r} / {args.lora_alpha} / {args.lora_dropout} |",
        f"| epochs / effective batch | {args.epochs} / {eff} |",
        f"| learning rate | {args.lr} |",
        f"| assistant-only loss | True |",
        f"| system prompt (sha256) | {_prompt_sha(system_prompt) if system_prompt else '-'} |",
        f"| final train loss | {train_loss:.4f} |",
        f"| best eval_loss | {best} |",
        f"| date | {datetime.now().isoformat(timespec='seconds')} |",
        f"| SLURM job | {os.environ.get('SLURM_JOB_ID', '-')} |",
        "",
        "See `train_log.md` for the full parameter list.",
        _prompt_section(system_prompt),
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote README to {out_dir / 'README.md'}", file=sys.stderr)


def main() -> int:
    args = parse_args()

    if not args.train_jsonl.exists():
        print(f"{args.train_jsonl} not found", file=sys.stderr)
        return 1

    if not args.max_seq_len:  # None or 0 -> fall back to the model's full context window
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

    class _LossOnlySFTTrainer(SFTTrainer):
        """SFTTrainer whose compute_loss returns the model's loss without touching
        ``outputs.logits``. The CCE forward returns ``logits=None`` (the whole point — the
        full seq×vocab logits are never built), which TRL's default compute_loss would crash
        on while computing its token-accuracy/entropy metric. Bypassing that metric is fine:
        CCE doesn't expose per-token logits anyway. This also frees us from setting
        ``use_liger_kernel=True`` (which would force a liger-kernel import) just to make TRL
        tolerate the missing logits."""
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs.pop("_prediction_loss_only", None)
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    data_files = {"train": str(args.train_jsonl)}
    if args.val_jsonl and args.val_jsonl.exists():
        data_files["validation"] = str(args.val_jsonl)
    dataset = load_dataset("json", data_files=data_files)
    print(f"train examples: {len(dataset['train'])}", file=sys.stderr)

    model, tokenizer = build_model_and_tokenizer(args)

    # Record the exact system prompt baked into the training data (for log/README),
    # and install the generation-tagged template so assistant_only_loss works.
    sample_messages = dataset["train"][0]["messages"]
    system_prompt = next((m["content"] for m in sample_messages
                          if m["role"] == "system"), "")
    enable_assistant_only_loss(tokenizer, sample_messages)

    # Guard the assistant_only_loss + CCE zero-token crash: drop records whose
    # conversation would be truncated past its assistant turn at this max_seq_len.
    dataset["train"] = drop_overlong_records(dataset["train"], tokenizer,
                                             args.max_seq_len, "train")
    if "validation" in dataset:
        dataset["validation"] = drop_overlong_records(dataset["validation"], tokenizer,
                                                       args.max_seq_len, "val")

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
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_length=args.max_seq_len,
        packing=args.packing,
        # Train on the assistant turn only (mask system+user). Relies on the
        # generation-tagged template installed by enable_assistant_only_loss().
        assistant_only_loss=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if has_val else "no",
        # Only eval_loss is used (early stopping + best model); no compute_metrics. With the
        # CCE fused-loss path the model returns logits=None, so the trainer must not gather
        # eval logits — and skipping them avoids a full seq×vocab eval OOM.
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

    # CCE returns logits=None, so use the loss-only trainer that skips TRL's logits metric.
    trainer_cls = _LossOnlySFTTrainer if args.cce else SFTTrainer
    trainer = trainer_cls(
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

    n_val = len(dataset["validation"]) if has_val else 0
    if log_path is not None:
        write_train_log(
            log_path, args, out_dir=out_dir, logging_dir=logging_dir,
            n_train=len(dataset["train"]), n_val=n_val,
            train_loss=result.training_loss,
            best_metric=trainer.state.best_metric,
            best_ckpt=trainer.state.best_model_checkpoint,
            system_prompt=system_prompt)
    # Human-readable run summary (replaces PEFT's generic model-card README.md).
    write_run_readme(out_dir, args, n_train=len(dataset["train"]), n_val=n_val,
                     train_loss=result.training_loss, best_metric=trainer.state.best_metric,
                     system_prompt=system_prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
