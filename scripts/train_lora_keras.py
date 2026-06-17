#!/usr/bin/env python3
"""Minimal Keras + JAX LoRA fine-tune — alternative to scripts/train_lora.py.

Approach #2 of branch ``fix/LoRA_alternative_implementations``: a *fundamentally
different stack*. KerasHub's ``GemmaCausalLM`` on the **JAX** backend has its own
autodiff and loss path, so it sidesteps the PyTorch cross-entropy NaN that broke
the liger/CCE attempts. Follows Google's official guide
(https://ai.google.dev/gemma/docs/core/lora_tuning): load a preset, call
``backbone.enable_lora(rank=...)``, ``fit()``.

Minimal = single device. Escalation if 31B doesn't fit one GPU: shard with
``keras.distribution.ModelParallel`` across all visible GPUs (see --model-parallel).

Reads the same chat-``messages`` JSONL as the other implementations and flattens
each record to one Gemma-chat-template string (Keras trains on plain text).

Open risk to resolve early: whether a gemma-4-31B KerasHub preset exists/converts.
If not, this documents the blocker; fall back to the largest available Gemma preset.

Needs an isolated env (JAX/Keras vs the torch stack) — run via `uv run --extra keras`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Backend must be selected before importing keras.
os.environ.setdefault("KERAS_BACKEND", "jax")
# JAX preallocates 75% of VRAM by default; let it grow so weights+activations coexist.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.0")

GEMMA_TEMPLATE = (
    "<start_of_turn>user\n{user}<end_of_turn>\n"
    "<start_of_turn>model\n{model}<end_of_turn>"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--val-jsonl", type=Path, default=None)
    p.add_argument("--preset", default="gemma3_instruct_1b",
                   help="KerasHub preset (default: small for smoke). Scale to the largest "
                        "Gemma-4 preset available, e.g. gemma3_instruct_27b / a gemma-4 preset.")
    p.add_argument("--out-dir", type=Path, default=Path("results/keras_lora"))
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="Keras preprocessor sequence_length (token cap incl. completion)")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--limit", type=int, default=None, help="Cap #records (smoke test)")
    p.add_argument("--model-parallel", action="store_true",
                   help="Shard the model across all visible GPUs with keras.distribution "
                        "(escalation for a preset too large for one GPU)")
    return p.parse_args()


def load_texts(path: Path, limit: int | None) -> list[str]:
    """Flatten chat-``messages`` records to Gemma-template strings (system folded
    into the user turn; loss is computed over the whole string in this minimal setup)."""
    texts: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            sys_txt = "".join(m["content"] for m in msgs if m["role"] == "system")
            user_txt = "".join(m["content"] for m in msgs if m["role"] == "user")
            model_txt = "".join(m["content"] for m in msgs if m["role"] == "assistant")
            user_full = (sys_txt + "\n\n" + user_txt).strip() if sys_txt else user_txt
            texts.append(GEMMA_TEMPLATE.format(user=user_full, model=model_txt))
            if limit and len(texts) >= limit:
                break
    return texts


def main() -> int:
    args = parse_args()
    if not args.train_jsonl.exists():
        print(f"{args.train_jsonl} not found", file=sys.stderr)
        return 1

    import keras
    import keras_hub

    if args.model_parallel:
        # Escalation path: tensor-parallel shard over every visible device.
        devices = keras.distribution.list_devices()
        print(f"model-parallel over {len(devices)} devices", file=sys.stderr)
        mesh = keras.distribution.DeviceMesh((len(devices),), ["model"], devices)
        layout_map = keras.distribution.LayoutMap(mesh)
        keras.distribution.set_distribution(
            keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="batch"))

    print(f"loading preset {args.preset}", file=sys.stderr, flush=True)
    gemma = keras_hub.models.GemmaCausalLM.from_preset(args.preset)
    gemma.backbone.enable_lora(rank=args.lora_rank)
    gemma.preprocessor.sequence_length = args.max_seq_len

    train_texts = load_texts(args.train_jsonl, args.limit)
    print(f"train examples: {len(train_texts)}", file=sys.stderr)

    gemma.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    gemma.fit(train_texts, epochs=args.epochs, batch_size=args.batch_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Save just the LoRA-adapted weights (small) for inference/comparison.
    gemma.backbone.save_lora_weights(str(args.out_dir / "lora.weights.h5"))
    print(f"saved LoRA weights to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
