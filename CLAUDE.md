# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Fork of Vision Transformer Hashing (VTS, ICME 2022) — a research codebase for deep image hashing with a ViT backbone, evaluated under multiple hashing frameworks (CSQ, DSH, DPN, HashNet, GreedyHash, IDHN). No test suite, no lint config — this is a research repo extended for coursework (master's IR lab).

## Commands

```bash
# Environment (venv at venv/, not uv-managed)
source venv/bin/activate
pip install -r requirements.txt          # torch, torchvision, numpy, scipy, matplotlib, Pillow, tqdm, ml-collections

# Train a hashing model (CLI args override get_config() inline)
python CSQ.py --dataset cifar10 --bit 16 --epoch 150 --test_map 30
python CSQ.py --dataset cifar10 --bit 32 --epoch 150 --backbone AlexNet --save_path Checkpoints_Results/CSQ-AlexNet-cifar10  # KB2
python DSH.py --dataset cifar10 --bit 16

# Build hash-code database from all trained checkpoints in train-models/
python features/build_database.py        # auto-discovers *.pth/.pt by filename pattern

# Search top-10 similar images for an uploaded query
python features/search_top10_uploaded.py # edit main() to set bit/model_path/query_image_path

# Colab KB5: open main_scenario_005.ipynb — CSQ + ViT-B_32, bit 16/32/64 song song
# Colab: open main.ipynb — sets up Drive symlinks to MyDrive/master_is/semester_3/IR/VTS-LAB/
```

Single-file runs are the unit of work; there is no test runner.

## Architecture

**Hashing algorithms** — one file per method, paired `<Method>.py` (hash output) and `<Method>cls.py` (classification head variant): `CSQ`, `DSH`, `DPN`, `HashNet`, `GreedyHash`, `IDHN`. Each defines its own `get_config()` dict (dataset, net, bit_list, optimizer, epochs, save_path) and a `train_val(config, bit)` loop. Loss class lives in the same file (e.g. `CSQLoss` in `CSQ.py`). To add a new method, copy the pattern — don't try to unify them.

**Backbones** — selected by uncommenting a line in `get_config()`:
- `network.py` — `AlexNet`, `ResNet` wrappers exposing a hash-bit head
- `TransformerModel/modeling.py` — `VisionTransformer` (loaded from `pretrainedVIT/ViT-B_{16,32}.npz`), configs in `vit_configs.py`. `modeling_cls.py` is the classification variant.

**Shared utilities** — `utils/tools.py` is the hub: `config_dataset()` (sets `n_class`, `topK` per dataset), `get_data()` (dataloaders for cifar10/coco/nuswide/imagenet), `compute_result()` (extract binary codes), `CalcTopMap()`, `pr_curve()`, `CalcHammingDist()`. CIFAR path is **relative** (`./dataset/...`) — do not reintroduce an absolute path.

**Checkpoints** — saved to `Checkpoints_Results/` as `{dataset}_{info}_{net_print}_Bit{bit}-{BestModel|IntermediateModel}.pth`. The `features/build_database.py` regex (`MODEL_FILENAME_RE`) depends on this exact filename schema — if you change naming in a training script, update the regex. For benchmark scenarios that compare frameworks, use subfolder convention `Checkpoints_Results/{framework}-{backbone}-{dataset}/` (e.g. `CSQ-ViT-B_32-cifar10/`) by passing `--save_path` — filename inside folder unchanged.

**Retrieval pipeline** (under `features/`):
1. `build_database.py` — iterates `train-models/*.pth`, parses filename → config, loads net, runs `compute_result()` on the dataset loader, writes per-model `database_index/{model_stem}/codes.npy + labels.npy + paths.txt`.
2. `search_top10_uploaded.py` — loads one model + its precomputed index, hashes a query image, ranks by Hamming distance, plots top-10 to `top10_uploaded_result.png`. Configuration is hardcoded in `main()` — bit / model_path / query_image_path must match an index built in step 1.

Detailed knowledge docs (read before non-trivial changes): `docs/ai/implementation/knowledge-csq.md`, `knowledge-build-database.md`, `knowledge-search-top10-uploaded.md`.

## Conventions specific to this repo

- Config is a plain `dict` returned by `get_config()` — not pydantic, not argparse-driven by default. CLI flags (`--dataset`, `--bit`, `--epoch`, `--test_map`, `--save_path`) were added on top to all 6 training scripts and only override a handful of fields; the dict is the source of truth.
- `device` in `get_config()` auto-falls back to CPU if CUDA unavailable — allows smoke testing locally (slow for ViT) without code changes.
- Checkpoints are `.pth` (current) — older `.pt` files exist in `train-models/` and the build_database regex accepts both. Do not rename existing files.
- `Checkpoints_Results/` is created on demand. The Colab notebook (`main.ipynb` cell 4) sets it up as a symlink into Drive — clean up stale symlinks before recreating to avoid `FileExistsError`.
- Backbone switching is by commenting/uncommenting lines in `get_config()`. There is no flag for it — accept that and don't refactor unless asked. **Exception: `CSQ.py` has `--backbone {AlexNet,ResNet,ViT-B_32,ViT-B_16}` CLI flag added for KB2 (Benchmark Scenario 002). All other framework files still use comment/uncomment.**
- `*cls.py` variants (CSQcls, DPNcls, DSHcls, GreedyHashcls, HashNetcls, IDHNcls) now have unified 5-flag CLI (`--dataset`, `--bit`, `--epoch`, `--test_map`, `--save_path`) added for KB3 (Benchmark Scenario 003). Backbone is fixed ViT-B_32 — no `--backbone` flag for cls variants. Output folder convention: `Checkpoints_Results/{Method}cls-ViT-B_32-{dataset}/`.
- Google Drive backup code has been intentionally removed from checkpoint-saving paths; do not reintroduce `shutil.copy` to Drive.

## Feedback language

Always reply in Vietnamese (per global `~/.claude/CLAUDE.md`).
