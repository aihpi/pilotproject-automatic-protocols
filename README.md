<div style="background-color: #ffffff; color: #000000; padding: 10px;">
<img src="00_aisc\img\logo_aisc_bmftr.jpg">
<h1> Automatic Protocols — committee transcripts → protocols
</div>

Fine-tune and run a LoRA adapter on `google/gemma-4` to turn Landtag/committee meeting
**transcripts** into structured German **protocols** (per agenda item / *Tagesordnungspunkt*).
The pipeline ingests raw audio/PDF/DOCX, prepares speaker-labelled and TOP-tagged transcripts,
builds an instruction dataset, trains a (Q)LoRA adapter, and generates protocol summaries.

## Pipeline

| stage | script | what it does |
|---|---|---|
| **A. Ingest** | `scripts/ingest_corpus.py`, `pdf_to_markdown.py`, `docx_to_markdown.py`, `transcribe.py` | stage the raw corpus; PDF/DOCX/audio → markdown |
| **B. Prepare** | `tag_transcript_tops.py`, `match_speakers.py` | tag agenda-item (TOP) boundaries and resolve speaker names → `data/transcripts/md_prepared/` |
| **C. Dataset** | `scripts/build_dataset.py` | pair transcripts↔protocols per TOP, write chat-format `train/val.jsonl` |
| **D. Train** | `scripts/train_lora.py` (+ `train_lora.sbatch`) | (Q)LoRA fine-tune on SLURM; long-context via cut-cross-entropy (`--cce`) |
| **E. Infer** | `scripts/infer_summary.py` (+ `infer_summary.sbatch`) | generate protocol summaries from new transcripts |

## Long-context training (the OOM fix)

gemma-4-31B has a ~262k-token vocabulary, so the standard loss materialises a
`seq_len × vocab` logits tensor that OOMs a single 80 GB H100 beyond ~32k tokens. Training
with **`--cce` (`USE_CCE=1`)** uses **cut-cross-entropy** to compute the loss without ever
building that tensor — enabling long and even uncapped sequences, numerically stable in bf16,
honouring Gemma's logit softcap. See `scripts/gemma4_cce_patch.py`.

## Quick start

```bash
# Train (SLURM) — long-context QLoRA on the 31B
USE_CCE=1 BITS=4 MAX_SEQ_LEN=32768 BASE_MODEL=google/gemma-4-31B-it \
  TRAIN_JSONL=data/train_no_docs/train.jsonl VAL_JSONL=data/train_no_docs/val.jsonl \
  sbatch --qos=aisc --gres=gpu:h100:2 --constraint=ARCH:X86 scripts/train_lora.sbatch

# Infer with a trained adapter
INPUT=data/transcripts/md_prepared/example_Transkript.md \
  ADAPTER=results/<YYYYMMDD-HHMMSS> sbatch scripts/infer_summary.sbatch
```

Each training run writes a self-contained `results/YYYYMMDD-HHMMSS/` folder (adapter + tokenizer
+ `train_log.md` + `README.md`). **Full pipeline details, flags, and SLURM/cluster notes are in
[`instructions.md`](instructions.md).**

## Limitations

- Built for German committee/Landtag protocols; the TOP-tagging heuristics are committee-oriented.
- Very long whole-document records (100k+ tokens) stress attention/activation memory even with
  cut-cross-entropy; cap or shard as needed.

## References

- [AI Service Centre Berlin-Brandenburg](https://hpi.de/kisz)

---

## Acknowledgements
<img src="00_aisc/img/logo_bmftr_de.png" alt="drawing" style="width:170px;"/>

The [AI Service Centre Berlin Brandenburg](http://hpi.de/kisz) is funded by the [Federal Ministry of Research, Technology and Space](https://www.bmbf.de/) under the funding code 16IS22092.
