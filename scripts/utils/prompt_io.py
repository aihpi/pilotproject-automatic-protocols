#!/usr/bin/env python3
"""Single source of truth for the summarization input contract.

Training (``build_dataset.py``), evaluation (``eval_io.py``) and inference
(``infer_summary.py`` / ``infer_unsloth.py``) all build their model input here so
it matches the deployment app byte-for-byte. The deployed summarizer
(``pilotproject-protokollierungsassistenz`` ``/api/summarize``) sends, PER TOP:

    system  = prompt_summarize.txt   (the ``gemma`` config's locked prompt)
    user    = build_user_message(top_title, transcript_text)

where ``transcript_text`` is clean ``Name: utterance`` lines (diarised, no
timestamps, no ``<SD-*>`` markers). This module renders our tagged source
transcripts into that exact shape.

Run ``python scripts/utils/prompt_io.py`` for the self-check.
"""

from __future__ import annotations

import re
from pathlib import Path

# Single source of truth for the system prompt: the file lives next to this module
# (scripts/utils/prompt_summarize.txt). No hardcoded fallback — a missing file is a
# loud error, not a silent drift from the deployment's locked prompt.
_PROMPT_FILE = Path(__file__).resolve().parent / "prompt_summarize.txt"

_SPK_RE = re.compile(r"<SD-SPK>(.*?)</SD>", re.DOTALL)
_TOPTAG_RE = re.compile(r"<SD-TOP>.*?</SD>", re.DOTALL)
# Leading "[00:00:00.420 --> 00:00:02.440] " timestamp on an utterance line.
_TS_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}(?:\.\d+)?\s*-->\s*\d{2}:\d{2}:\d{2}(?:\.\d+)?\]\s*")


def load_summary_prompt() -> str:
    """The summarization system prompt, read from ``scripts/utils/prompt_summarize.txt``.

    Raises ``FileNotFoundError`` with a clear message if the file is missing — the
    prompt is the single source of truth shared by training, evaluation and
    inference, so silently substituting anything else would hide drift."""
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise FileNotFoundError(
            f"summary system prompt not found at {_PROMPT_FILE}: {exc}. "
            "It is the single source of truth for training/eval/inference — "
            "restore the file (do not substitute a fallback)."
        ) from exc


def render_transcript_text(segment: str) -> str:
    """Render one source TOP segment (``<SD-SPK>Name</SD>`` turns + ``[ts] text``
    lines) into the deployment's clean ``Name: utterance`` lines. Consecutive
    utterances of the same speaker are merged into ONE space-joined line — exactly
    what the deployment's transcribe.py does (``transcript[-1]["text"] += " " + text``)
    — so train/eval input matches serve. Timestamps and ``<SD-TOP>``/``<SD-SPK>``
    markers are stripped; text before any speaker tag is emitted without a prefix."""
    turns: list[list[str]] = []  # [speaker, merged_text]
    speaker = ""
    for raw in _TOPTAG_RE.sub("", segment).splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SPK_RE.match(line)
        if m:
            speaker = m.group(1).strip()
            continue
        utt = _TS_RE.sub("", line).strip()
        if not utt:
            continue
        if turns and turns[-1][0] == speaker:
            turns[-1][1] += " " + utt
        else:
            turns.append([speaker, utt])
    return "\n".join(f"{spk}: {txt}" if spk else txt for spk, txt in turns)


def build_user_message(top_title: str, transcript_text: str) -> str:
    """The exact ``/api/summarize`` user message (see summarize.py)."""
    return (
        "Erstelle eine Zusammenfassung für folgenden Tagesordnungspunkt:\n\n"
        f"TOP: {top_title}\n\n"
        f"Transkript:\n{transcript_text}\n\n"
        "Zusammenfassung:"
    )


def top_title_from_protocol(protocol: str, n: int) -> str:
    """Clean agenda title for TOP ``n`` from the gold protocol's *Aus der Beratung*
    heading ``## Zu TOP N <title>``. The *Beschlüsse* pass heading ``## Zu TOP N:``
    (colon) carries the decision, not a title, and is excluded by the negative
    lookahead — so this never leaks the answer into the input. Whitespace is
    collapsed (PDF-extraction spacing). Falls back to ``f"TOP {n}"``.

    Used for TRAINING input only: the gold heading is the agenda topic (the same
    text the target output starts with), so the model learns to USE the title
    rather than invent it. At serve the title comes from the app's invitation
    extraction; eval/inference (no protocol available) use the ``f"TOP {n}"`` fallback."""
    m = re.search(rf"(?im)^#*\s*Zu\s+TOP\s*{n}(?![\d:.])\s+(.+)$", protocol)
    if not m:
        return f"TOP {n}"
    return re.sub(r"\s+", " ", m.group(1)).strip(" :-–") or f"TOP {n}"


def _selfcheck() -> None:
    seg = (
        "<SD-TOP>TOP 3</SD>\n"
        "<SD-SPK>Annemarie Wolff (SPD)</SD>\n"
        "[00:00:00.420 --> 00:00:02.440] Guten Morgen.\n"
        "[00:00:04.820 --> 00:00:12.140] Ich begrüße die Mitglieder.\n"
        "<SD-SPK>Steffen Freiberg</SD>\n"
        "[00:01:00.000 --> 00:01:02.000] Vielen Dank.\n"
    )
    rendered = render_transcript_text(seg)
    # consecutive same-speaker utterances merged into one space-joined line (option b)
    assert rendered == (
        "Annemarie Wolff (SPD): Guten Morgen. Ich begrüße die Mitglieder.\n"
        "Steffen Freiberg: Vielen Dank."
    ), rendered
    assert "<SD-" not in rendered and "[00:" not in rendered

    user = build_user_message("TOP 3", rendered)
    assert user.startswith("Erstelle eine Zusammenfassung für folgenden Tagesordnungspunkt:\n\nTOP: TOP 3\n\nTranskript:\n")
    assert user.endswith("\n\nZusammenfassung:")

    pr = ("## Beschlüsse und Festlegungen:\n## Zu TOP 2:\nDer Ausschuss beschließt einstimmig (9 : 0 : 0).\n"
          "## Aus der Beratung:\n## Zu TOP 2 Gesetz zu  dem  Abkommen\nBeratung …")
    assert top_title_from_protocol(pr, 2) == "Gesetz zu dem Abkommen", top_title_from_protocol(pr, 2)
    assert top_title_from_protocol("## Zu TOP 5:\nBeschluss.", 5) == "TOP 5"
    assert top_title_from_protocol("(kein Titel)", 7) == "TOP 7"

    assert load_summary_prompt().startswith("Du bist Protokollführer")
    print("prompt_io self-check: OK")


if __name__ == "__main__":
    _selfcheck()
