#!/usr/bin/env python3
"""Small shared helpers for the LoRA train/build/infer scripts.

Kept dependency-light: transformers is imported lazily inside the function that
needs it, so importing this module stays cheap.
"""
from __future__ import annotations


def context_window(base_model: str, default: int = 4096) -> int:
    """Return the model's max context length (tokens) from its HF config.

    Gemma-4 is multimodal, so the text limit lives under ``config.text_config``
    (gemma-4-31B-it = 262144, gemma-4-E2B-it = 131072). Falls back to ``default``
    if the field is absent.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(base_model)
    sub = getattr(cfg, "text_config", None) or cfg
    n = (getattr(sub, "max_position_embeddings", None)
         or getattr(cfg, "max_position_embeddings", None))
    return int(n) if n else default
