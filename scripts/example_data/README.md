# Example data — public plenary sessions

This folder builds a small, **self-contained example dataset** from public Landtag Brandenburg plenary sessions so the LoRA pipeline can be run end-to-end without the internal corpus.

> The example data below is **rebuilt on demand** rather than committed (`data_example/` is gitignored — the videos are multi-GB). Run the steps here to (re)create it.
>
> Find more material to add:
> - Videos — Mediathek: <https://www.landtag.brandenburg.de/de/aktuelles/neuigkeiten/mediathek/>
> - Protocols — Parlamentsdokumentation: <https://www.parlamentsdokumentation.brandenburg.de/portal/browse.tt.html>


## The video ↔ protocol link

The connection between a video and its protocol is the **plenary sitting number** *N* (8. Wahlperiode):

- **Protocol** — a born-digital PDF at a fixed URL pattern: `…/parladoku/w8/plpr/<N>.pdf` (`plpr` = Plenarprotokoll, `w8` = 8. Wahlperiode).
- **Video** — a Mediathek page that embeds a [3Q SDN](https://www.3qsdn.com/) player as `data-provider="threeqsdn"` / `data-dataid="<uuid>"`. The stream resolves via `https://playout.3qsdn.com/<dataid>`, which `yt-dlp`'s built-in `3qsdn` extractor downloads.

All plenary sittings are public (*öffentlich*).

> **Note — committee vs. plenary.** Committee sessions (*Ausschusssitzungen*) are a *separate* system: their protocols are *Ausschussprotokolle* under a different path (`…/parladoku/w8/apr/…`) and many are *nicht öffentlich*. The example set here is **plenary** only, where the sitting number gives a clean 1:1 video↔protocol pairing.

### `manifest.tsv` — the source of truth

One row per sitting: `number<TAB>mediathek_url` (`#` comments and blanks ignored). The protocol URL is derived from the number. To add a pair, copy the plenary-sitting page URL from the Mediathek and add a row with its sitting number.


## End-to-end pipeline on the example data

All scripts take `--input`/`--out-dir`, skip existing outputs unless `--overwrite`, and use exit codes 0/1/2 — same conventions as the main pipeline (`../`). Install deps once with `uv sync` (adds `yt-dlp`, `pyannote.audio`, `torchaudio`).

```bash
# 0. (one-time) HuggingFace token for the gated diariser — see "Diarisation" below.
cp .env_example .env && $EDITOR .env        # set HF_TOKEN=…

# 1. Download videos (mp4, lowest resolution) + protocols (pdf) into data_example/
uv run python scripts/example_data/download.py --out-dir data_example
#   protocols only (fast, no multi-GB video):  … download.py --skip-video
#   one sitting:                                … download.py --only 31

# 2. Extract 16 kHz mono WAV from each mp4 (what the transcriber consumes)
uv run python scripts/example_data/extract_audio.py \
    --input data_example/transcripts/mp4 --out-dir data_example/transcripts/wav

# 3. Transcribe + diarise the audio → markdown with <SD-SPK> speaker turns (GPU)
ls data_example/transcripts/wav/*.wav > data_example/wav_manifest.txt
DIARIZE=1 INPUT_LIST=data_example/wav_manifest.txt \
    OUT_DIR=data_example/transcripts/md sbatch scripts/transcribe.sbatch
#   locally:  uv run python scripts/transcribe.py --diarize \
#                 --input-list data_example/wav_manifest.txt \
#                 --out-dir data_example/transcripts/md

# 4. Protocol PDF → clean markdown (drops the title/Inhalt header, "Drucksache …"
#    references and hyperlinks; starts the body at "Beginn der Sitzung")
uv run python scripts/example_data/protocol_pdf_to_markdown.py \
    --input data_example/protocols/pdf --out-dir data_example/protocols/md_clean

# 5. Match speakers across both files and replace real names with consistent
#    generic tags (SPEAKER_00 …) on BOTH sides — see "Speaker matching" below.
uv run python scripts/example_data/match_speakers.py \
    --transcript-dir data_example/transcripts/md \
    --protocol-dir   data_example/protocols/md_clean
#   -> writes data_example/transcripts/md_anon, data_example/protocols/md_anon,
#      and a per-session mapping report under data_example/speaker_maps/

# 6. Pair transcripts ↔ protocols → chat JSONL (use the *anonymised* dirs).
#    Plenary protocols have no "Zu TOP N" markers, so use document granularity.
uv run python scripts/build_dataset.py \
    --transcript-dir data_example/transcripts/md_anon \
    --protocol-dir   data_example/protocols/md_anon \
    --granularity document --out-dir data_example/train

# 7. Train the LoRA adapter on the example JSONL (GPU / SLURM)
TRAIN_JSONL=data_example/train/train.jsonl sbatch scripts/train_lora.sbatch

# 8. Inference on a sample transcript
INPUT=data_example/transcripts/md_anon/Plenum_8-31_Transkript.md \
    ADAPTER=results/lora_adapter sbatch scripts/infer_summary.sbatch
```

Filenames are kept consistent (`Plenum_8-<N>_Transkript.*` / `Plenum_8-<N>_Protokoll.*`) so `build_dataset.normalise_stem` pairs them automatically.

> The example set is tiny (a handful of sessions) — enough to exercise the *plumbing*, not to train a useful adapter. For real training, scale up via the internal corpus or by adding many rows to `manifest.tsv`.


## Diarisation

`transcribe.py --diarize` runs [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) on the waveform and assigns each Whisper segment to the speaker turn it most overlaps, grouping consecutive same-speaker segments under a `<SD-SPK>SPEAKER_00</SD>` header.

The models are **gated** — request/accept access on HuggingFace (while logged in) and set `HF_TOKEN` in `.env` (the sbatch sources it):
- <https://huggingface.co/pyannote/speaker-diarization-community-1> — the pipeline (4.x); "Ask for access" and wait for approval.
- <https://huggingface.co/pyannote/segmentation-3.0> — the segmentation model it loads.

We require `pyannote.audio>=4` because 3.x calls `torchaudio.AudioMetaData`, removed in the torchaudio 2.11 that matches this repo's torch 2.11. The older `pyannote/speaker-diarization-3.1` id still works as `--diarization-model`, but under 4.x it pulls the same `community-1` PLDA, so access to `community-1` is needed either way.

Optional bounds: `--num-speakers`, `--min-speakers`, `--max-speakers` (sbatch: `NUM_SPEAKERS=…`). Without `--diarize` the transcriber output is unchanged (flat segment lines, no speaker headers).


## Speaker matching — `match_speakers.py`

Diarisation produces **anonymous** labels (`SPEAKER_00`, `SPEAKER_01`, …) that are unrelated to the **real names** in the protocol (`## Dr. Dietmar Woidke (Ministerpräsident):`). Training on `SPEAKER_00` → real-name pairs gives the model no learnable bridge and yields protocols with invented names. `match_speakers.py` fixes this by giving **every person one consistent generic tag in both files** (transcript `<SD-SPK>SPEAKER_XX</SD>`, protocol `## SPEAKER_XX:`), indexed by first appearance in the protocol.

It maps each pyannote label to a canonical protocol speaker via a **priority cascade** (a label resolved by a higher method is never overridden):

1. **Explicit chair naming** — the chair announces the next speaker inside their own turn ("… darf ich Herrn Ministerpräsidenten Dr. *Woidke* bitten", "Ich sehe Herrn Abgeordneten *Ossowski* am Mikrofon"); the label of the *following* turn is voted to that surname.
2. **Content matching** — for still-unmapped labels, fuzzy-match the label's utterances against each protocol speaker's text (rapidfuzz `token_set_ratio`, `--content-threshold`, default 60).
3. **Rednerliste / sequence** — whatever remains is aligned by chronological position of transcript turns vs. protocol headings.

On the example session this resolved 42 pyannote labels via **21 chair / 20 content / 1 sequence**. A per-session report (`data_example/speaker_maps/<stem>.json`) records each label → name → `SPEAKER_XX` + method + score for inspection. Other options considered but not used here: a *roster in the prompt*, *voice enrollment* against known MdL embeddings, or *LLM post-attribution*.

**Scope / caveat:** only the speaker **attribution** is anonymised (the `<SD-SPK>` tags and `## Name:` headings). Names *spoken within* utterances or appearing in protocol prose/interjections (e.g. `(Lena Kotré [AfD]: …)`) are **left intact** — scrubbing every in-text mention is a separate, riskier pass. Matching is best-effort: ASR misspellings, pyannote over/under-segmentation, and noisy pre-session audio cause occasional mislabels (visible in the report's low-confidence rows).
