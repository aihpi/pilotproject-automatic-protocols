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

# KerasHub's Gemma instruction preprocessor expects a dict of {prompts, responses}
# (it adds the turn markers itself); it indexes x["prompts"], so flat strings fail.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--val-jsonl", type=Path, default=None)
    p.add_argument("--preset", default="hf://google/gemma-3-1b-it",
                   help="KerasHub preset. Default uses the HF transformers->Keras converter "
                        "(no Kaggle auth). Kaggle handles (e.g. gemma3_instruct_1b) need "
                        "KAGGLE_USERNAME/KAGGLE_KEY. gemma-4 may lack a KerasHub converter "
                        "(documented risk) — fall back to the largest convertible Gemma.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Run folder. Default: auto-named results/YYYYMMDD-HHMMSS")
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


def load_pairs(path: Path, limit: int | None) -> dict[str, list[str]]:
    """Chat-``messages`` records -> {"prompts": [...], "responses": [...]} for the
    KerasHub Gemma instruction preprocessor (system folded into the user prompt)."""
    prompts: list[str] = []
    responses: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            sys_txt = "".join(m["content"] for m in msgs if m["role"] == "system")
            user_txt = "".join(m["content"] for m in msgs if m["role"] == "user")
            model_txt = "".join(m["content"] for m in msgs if m["role"] == "assistant")
            prompts.append((sys_txt + "\n\n" + user_txt).strip() if sys_txt else user_txt)
            responses.append(model_txt)
            if limit and len(prompts) >= limit:
                break
    return {"prompts": prompts, "responses": responses}


def main() -> int:
    args = parse_args()
    if not args.train_jsonl.exists():
        print(f"{args.train_jsonl} not found", file=sys.stderr)
        return 1

    import keras
    import keras_hub
    import tensorflow as tf
    # Keras runs on JAX; keep TensorFlow (used by the tf.data preprocessing) off the
    # GPU so it doesn't try to grab all 80 GB and OOM against JAX.
    tf.config.set_visible_devices([], "GPU")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # ponytail: shared helpers stayed in scripts/
    from alt_utils import resolve_out_dir, write_run_log, write_run_readme

    args.out_dir = resolve_out_dir(args.out_dir)

    if args.model_parallel:
        # Tensor-parallel shard the gemma-4 text decoder over all visible GPUs (the
        # 31B doesn't fit one). 2D mesh (batch=1, model=N); batch_dim_name="batch".
        # Rules target the text-decoder weight paths (verified from Gemma4Backbone):
        # shard query heads / attention_output / ffw / both token-embeddings on the
        # model axis; KV is GQA (1 head) so it stays replicated (no rule). Vision/audio
        # encoders are left replicated — irrelevant to text-only LoRA.
        devices = keras.distribution.list_devices()
        print(f"model-parallel over {len(devices)} devices", file=sys.stderr)
        mesh = keras.distribution.DeviceMesh((1, len(devices)), ["batch", "model"], devices)
        lm = keras.distribution.LayoutMap(mesh)
        lm["token_embedding/embeddings"] = ("model", None)
        lm["per_layer_token_embedding/embeddings"] = ("model", None)
        lm["per_layer_model_projection/kernel"] = (None, "model")
        lm["decoder_block.*attention/query/kernel"] = ("model", None, None)
        lm["decoder_block.*attention/attention_output/kernel"] = ("model", None, None)
        lm["decoder_block.*ffw_gating.*/kernel"] = (None, "model")
        lm["decoder_block.*ffw_linear/kernel"] = ("model", None)
        lm["decoder_block.*per_layer_input_gate/kernel"] = (None, "model")
        lm["decoder_block.*per_layer_up_proj/kernel"] = ("model", None)
        keras.distribution.set_distribution(
            keras.distribution.ModelParallel(layout_map=lm, batch_dim_name="batch"))

    print(f"loading preset {args.preset}", file=sys.stderr, flush=True)
    # The auto-dispatching `CausalLM.from_preset` mis-resolves gemma-4 "-it" presets to
    # the (incompletely-constructable) Gemma4AssistantCausalLM, so pick Gemma4CausalLM
    # explicitly for gemma-4; keep the auto class for gemma-3 etc.
    p = args.preset.lower()
    ModelCls = (keras_hub.models.Gemma4CausalLM
                if ("gemma-4" in p or "gemma4" in p) else keras_hub.models.CausalLM)
    gemma = ModelCls.from_preset(args.preset)
    gemma.backbone.enable_lora(rank=args.lora_rank)
    gemma.preprocessor.sequence_length = args.max_seq_len

    train_data = load_pairs(args.train_jsonl, args.limit)
    print(f"train examples: {len(train_data['prompts'])}", file=sys.stderr)

    gemma.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay),
        # No weighted_metrics under model-parallel: the accuracy metric's argmax/compare
        # triggers a host-side bool() on a sharded array -> NCCL allReduce failure.
        weighted_metrics=None if args.model_parallel else [keras.metrics.SparseCategoricalAccuracy()],
    )
    gemma.fit(x=train_data, epochs=args.epochs, batch_size=args.batch_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Save just the LoRA-adapted weights (small) for inference/comparison.
    gemma.backbone.save_lora_weights(str(args.out_dir / "model.lora.h5"))  # KerasHub requires *.lora.h5
    write_run_log(args.out_dir, "keras-jax", [
        ("preset", args.preset),
        ("max seq len", args.max_seq_len),
        ("LoRA rank", args.lora_rank),
        ("epochs", args.epochs),
        ("batch size", args.batch_size),
        ("model parallel", args.model_parallel),
        ("train jsonl", str(args.train_jsonl)),
    ])
    write_run_readme(args.out_dir, "keras-jax", args.preset,
                     f"KerasHub LoRA (rank {args.lora_rank}), JAX backend, seq_len {args.max_seq_len}. "
                     "Weights saved as lora.weights.h5 (NOT a PEFT adapter; needs a Keras loader).", [
                         ("preset", args.preset),
                         ("LoRA rank", args.lora_rank),
                         ("max seq len", args.max_seq_len),
                         ("epochs", args.epochs),
                         ("model parallel", args.model_parallel),
                     ])
    print(f"saved LoRA weights to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
