#!/bin/bash
#SBATCH --job-name=pdf_to_md
#SBATCH --account=aisc                        # AISC account for resource access
#SBATCH --partition=aisc-batch               # AISC nodes; CPU-only job (no --gres GPU requested)
#SBATCH --constraint=ARCH:X86               # the uv binary is x86-only
#SBATCH --exclude=gx13v1                     # known-flaky node
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=48G                            # Docling layout models are memory-hungry
#SBATCH --time=08:00:00
#SBATCH --output=logs/pdf_to_md_%j.out
#SBATCH --error=logs/pdf_to_md_%j.err
#
# Convert protocol PDFs -> markdown on a compute node (the login node OOM-kills
# large batches). Idempotent: pdf_to_markdown.py skips PDFs whose .md exists, so
# resubmitting resumes where a previous run stopped.
#
# Launch:  INPUT=data/protocols/pdf OUT_DIR=data/protocols/md sbatch scripts/pdf_to_markdown.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [ -f .env ]; then set -a; . ./.env; set +a; fi

INPUT="${INPUT:-data/protocols/pdf}"
OUT_DIR="${OUT_DIR:-data/protocols/md}"

export PATH="$HOME/.local/bin:$PATH"

uv run python scripts/pdf_to_markdown.py --input "$INPUT" --out-dir "$OUT_DIR"
