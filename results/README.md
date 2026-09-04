# Training runs

`results/` is gitignored; this file indexes the run folders on the cluster. Each run folder holds `adapter_config.json`, `adapter_model.safetensors`, tokenizer files, `train_log.md` (hyperparameters, dataset path, per-epoch losses) and TensorBoard logs under `runs/`.

## Current runs (the cap sweep, September 2026)

| run | job | base | dataset | cap | LoRA | result |
|---|---|---|---|---|---|---|
| `20260901-31b_cap32k` | 2501221 | gemma-4-31B-it, 4-bit | `data/train/cap32k` | 32,768 | r8 / a8 | 3 epochs, eval 0.6936 |
| `20260901-31b_cap40k` | 2501555 | gemma-4-31B-it, 4-bit | `data/train/cap40k` | 40,960 | r8 / a8 | 3 epochs, eval 0.6942 |
| **`20260902-31b_cap48k`** | 2502986 | gemma-4-31B-it, 4-bit | `data/train/cap48k` | 49,152 | r8 / a8 | **3 epochs, eval 0.6927; in production as `gemma-4-31b-protokoll`** |
| `20260902-31b_cap64k` | 2502987 | gemma-4-31B-it, 4-bit | `data/train/cap65k` | 65,536 | r8 / a8 | CUDA OOM at step 96 (logits tensor of a ~56k-token record) |

All four: `scripts/train_lora_unsloth.py` via `train_lora_unsloth.sbatch`, one H100 80 GB, lr 2e-4, batch 1 x 4 accumulation, warmup 5, weight decay 0.001, early stopping patience 3.

## Deploying a run

The hub accepts `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`, `tokenizer_config.json` and `README.md` in a flat tarball; `base_model_name_or_path` must be set to `google/gemma-4-31B-it` (the hub serves the fp16 base). The procedure and the hub API are documented in the app repository (`docs/deploy-lora-adapter.md`).

## Archive

`results/archive/` holds the 44 earlier runs and the shared `tensorboard/` folder, plus the old overview tables (`OVERVIEW.md`, `OVERVIEW_tllm.tsv`, `OVERVIEW_wp7.tsv`). Two entries matter:

- `archive/20260622-202658-legacy`: the adapter that was in production before the cap48k run (r32 / a32, cap 65,536, 229 / 22 records from 68 sittings, eval 0.7369). Kept for rollback.
- `archive/20260626-115353`: the 583 / 74 cap65k run on the WP7 superset (job 2293906).

The remaining archived runs are 2026-06 experiments with smaller caps, other frameworks and smoke tests. The overview tables also list runs that never produced a folder (`20260626-0943xx`, `20260626-115355`); those rows are dead.
