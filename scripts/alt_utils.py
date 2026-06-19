#!/usr/bin/env python3
"""Shared helpers for the alternative LoRA training scripts (Unsloth/Keras/FSDP).

Keeps result folders consistent with the main pipeline: every run is a
self-contained, timestamp-named folder ``results/YYYYMMDD-HHMMSS/`` holding the
adapter/weights, tokenizer and a ``train_log.md`` (the framework, base model and
dataset live in the log content, not the folder name).
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def to_convo(messages: list[dict]) -> list[dict]:
    """Fold a chat-`messages` record into a 2-turn Gemma conversation with content
    as typed parts. Gemma-4's chat template requires list-of-parts content and folds
    the system prompt into the user turn (it rejects a standalone system role)."""
    sys_txt = "".join(m["content"] for m in messages if m["role"] == "system")
    user_txt = "".join(m["content"] for m in messages if m["role"] == "user")
    model_txt = "".join(m["content"] for m in messages if m["role"] == "assistant")
    user_full = (sys_txt + "\n\n" + user_txt).strip() if sys_txt else user_txt
    return [
        {"role": "user", "content": [{"type": "text", "text": user_full}]},
        {"role": "model", "content": [{"type": "text", "text": model_txt}]},
    ]


def render_chat(tokenizer, messages: list[dict]) -> str:
    """Render a chat-`messages` record to a single training string via the Gemma
    template (loss over the whole string; mask the prompt only with a
    {% generation %}-aware template + assistant_only_loss)."""
    return tokenizer.apply_chat_template(to_convo(messages), tokenize=False,
                                         add_generation_prompt=False)


def resolve_out_dir(out_dir: Path | None) -> Path:
    """Return the run folder. Explicit --out-dir wins; otherwise auto-name
    ``results/YYYYMMDD-HHMMSS`` (matching scripts/train_lora.py)."""
    if out_dir is not None:
        return out_dir
    return Path("results") / datetime.now().strftime("%Y%m%d-%H%M%S")


def _prompt_sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _prompt_section(system_prompt: str) -> str:
    if not system_prompt:
        return "\n## System prompt\n\n_(none found in training data)_\n"
    return (f"\n## System prompt\n\nsha256: `{_prompt_sha(system_prompt)}`\n\n"
            f"```\n{system_prompt}\n```\n")


def write_run_log(out_dir: Path, framework: str, rows: list[tuple[str, object]],
                  system_prompt: str = "") -> None:
    """Write ``out_dir/train_log.md`` (same shape as the main pipeline's log)."""
    base = [
        ("framework", framework),
        ("date", datetime.now().isoformat(timespec="seconds")),
        ("SLURM job", os.environ.get("SLURM_JOB_ID", "-")),
    ]
    extra = [("system prompt (sha256)", _prompt_sha(system_prompt) if system_prompt else "-")]
    lines = [f"# Training run — {out_dir.name}", "",
             "| Parameter | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in base + rows + extra]
    lines.append(_prompt_section(system_prompt))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_readme(out_dir: Path, framework: str, base_model: str,
                     summary: str, rows: list[tuple[str, object]],
                     system_prompt: str = "") -> None:
    """Write a human-readable ``README.md`` into the run dir, mirroring
    ``scripts/train_lora.py::write_run_readme`` (overwrites PEFT's generic
    model-card README). Companion to the full ``train_log.md``."""
    lines = [
        f"# LoRA adapter — {out_dir.name}",
        "",
        f"Fine-tuned `{base_model}` via **{framework}**. {summary}",
        "",
        "| setting | value |",
        "|---|---|",
        f"| framework | {framework} |",
        f"| base model | `{base_model}` |",
    ]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += [
        f"| system prompt (sha256) | {_prompt_sha(system_prompt) if system_prompt else '-'} |",
        f"| date | {datetime.now().isoformat(timespec='seconds')} |",
        f"| SLURM job | {os.environ.get('SLURM_JOB_ID', '-')} |",
        "",
        "See `train_log.md` for the full parameter list.",
        _prompt_section(system_prompt),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
