#!/bin/bash
# =============================================================================
# Title:   Stage A — convert protocol PDFs to markdown (Docling)
# Purpose: Run scripts/pdf_to_markdown.py on a compute node (the login node OOM-kills
#          large batches). Idempotent: PDFs whose .md exists are skipped.
# Cluster: written for the HPI sc cluster (SLURM). Every "# ADJUST:" line must be
#          changed for a different SLURM cluster.
# Submit:  INPUT=data/protocols/pdf OUT_DIR=data/protocols/md sbatch scripts/pdf_to_markdown.sbatch
# =============================================================================

#SBATCH --job-name=pdf_to_md
#SBATCH --account=aisc                        # ADJUST: cluster account
#SBATCH --qos=aisc                            # ADJUST: QOS for the account
#SBATCH --partition=aisc-batch                # ADJUST: CPU-only job (no --gres); only aisc-batch accepts sbatch here
#SBATCH --constraint=ARCH:X86                 # ADJUST: the uv binary is x86-only
#SBATCH --exclude=gx13v1                      # ADJUST: known-flaky node
#SBATCH --cpus-per-task=8                     # ADJUST: CPUs
#SBATCH --nodes=1
#SBATCH --mem=48G                             # ADJUST: RAM (Docling layout models)
#SBATCH --time=08:00:00                       # ADJUST: walltime
#SBATCH --output=logs/pdf_to_md_%j.out
#SBATCH --error=logs/pdf_to_md_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [ -f .env ]; then set -a; . ./.env; set +a; fi

INPUT="${INPUT:-data/protocols/pdf}"
OUT_DIR="${OUT_DIR:-data/protocols/md}"

export PATH="$HOME/.local/bin:$PATH"

uv run python scripts/pdf_to_markdown.py --input "$INPUT" --out-dir "$OUT_DIR"
