#!/usr/bin/env python3
"""Small shared helpers for the LoRA train/build/infer scripts.

Kept dependency-light: transformers is imported lazily inside the function that
needs it, so importing this module stays cheap.
"""
from __future__ import annotations

import re
from pathlib import Path


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


def next_adapter_stamp(results_dir: Path, date_str: str) -> str:
    """Next ``YYYYMMDD_XX`` stamp for today, scanning ``results/`` for existing
    ``{date}_XX_lora`` adapters. XX runs 00..99 (starts at 00)."""
    rx = re.compile(rf"^{re.escape(date_str)}_(\d{{2}})_lora$")
    used = -1
    if results_dir.is_dir():
        for p in results_dir.iterdir():
            m = rx.match(p.name)
            if m:
                used = max(used, int(m.group(1)))
    return f"{date_str}_{used + 1:02d}"
