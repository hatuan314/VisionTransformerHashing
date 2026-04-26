# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Vision Transformer Hashing (VTS), IEEE ICME 2022 — PyTorch implementation that plugs a ViT backbone into six deep hashing frameworks (DSH, HashNet, GreedyHash, IDHN, CSQ, DPN) for image retrieval. Built on top of [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) and [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch).

## Running experiments

Each method is a standalone script that trains over `bit_list` (default `[64, 32, 16]`) sequentially:

```bash
python DSH.py        # also: HashNet.py, GreedyHash.py, IDHN.py, CSQ.py, DPN.py
python DSHcls.py     # *cls.py variants — same methods using the ViT [CLS] token head
```

`main.ipynb` runs the same scripts from a notebook. There are no tests, no lint config, and no build step.

### Required setup before training

- Pretrained ViT weights must exist at `pretrainedVIT/ViT-B_16.npz` and/or `pretrainedVIT/ViT-B_32.npz` (download from the `vit_models/imagenet21k` GCS bucket).
- Datasets live under `data/` (`cifar10`, `coco`, `nuswide_21`, `imagenet`). `data/coco` and `data/nuswide_21` ship with index lists; CIFAR-10 is downloaded automatically by `utils/tools.py` on first run.
- A CUDA GPU is assumed — `device` is hard-coded to `torch.device("cuda")` in every `get_config()`.
- Checkpoints, result logs, and PR-curve numbers are written to `Checkpoints_Results/` as `{dataset}_{info}_{net_print}_Bit{bit}-{BestModel|IntermediateModel}.pt` and `.txt`. Training auto-resumes from `*-IntermediateModel.pt` if present.

### Switching configuration

There is no CLI — edit `get_config()` at the top of the method's `.py` file. Toggle commented lines to:

- pick **dataset**: `cifar10` / `cifar10-2` / `coco` / `nuswide_21` / `imagenet` (this drives `n_class` and `topK` via `utils.tools.config_dataset`),
- pick **backbone**: `AlexNet`, `ResNet`, or `VisionTransformer` with `model_type`/`pretrained_dir` set to `ViT-B_16` or `ViT-B_32`,
- change **hash bits** via `bit_list`,
- adjust `epoch`, `test_map` (eval cadence), `batch_size`, `crop_size`, and method-specific knobs (e.g. `alpha`).

`precision_recall_curve.py` plots PR data emitted into the per-run `.txt` log.

## Architecture

The repo is intentionally flat — one file per (method × head-style) combination — so the high-level pattern matters more than the file list:

- **Method scripts (`{Method}.py`, `{Method}cls.py`)** — each defines `get_config()`, `train_val(config, bit)`, and a `*Loss` `nn.Module`. They share the same training skeleton: build loaders via `get_data`, build net (ViT branch passes `vit_config, crop_size, zero_head, num_classes, hash_bit`; CNN branch just passes `bit`), load pretrained ViT via `net.load_from(np.load(...))` only on a fresh start, train with Adam, evaluate every `test_map` epochs with `CalcTopMap` + `pr_curve`, persist best/intermediate checkpoints. To add or modify a method, mirror this structure rather than introducing a new abstraction.
- **`network.py`** — CNN backbones (`AlexNet`, `ResNet`) ending in a `hash_layer` that emits `bit`-dim codes.
- **`TransformerModel/`** — vendored ViT.
  - `modeling.py` (used by `{Method}.py`) attaches a hashing head that pools patch tokens.
  - `modeling_cls.py` (used by `{Method}cls.py`) attaches the head on the `[CLS]` token instead — this is the only difference between `Foo.py` and `Foocls.py` pairs.
  - `vit_configs.py` exposes `VIT_CONFIGS` keyed by `"ViT-B_16"` / `"ViT-B_32"`; `modeling_resnet.py` is the hybrid ResNet stem from the original ViT repo.
- **`utils/tools.py`** — central data plumbing: `config_dataset(config)` injects `topK` and `n_class` per dataset, `get_data(config)` returns `(train_loader, test_loader, dataset_loader, ...)` where `dataset_loader` is the retrieval gallery, `compute_result` extracts binary codes, `CalcTopMap` computes mAP@topK, `pr_curve` produces PR arrays. All method scripts depend on these names via `from utils.tools import *`.

### Conventions worth respecting when editing

- The tuple returned by the dataloader is `(image, label, ind)` — `ind` is the global index used by methods like DSH/HashNet/IDHN to maintain a running buffer (`self.U`, `self.Y`) of size `num_train × bit`. Preserve this contract when adding a new method or loss.
- `"ViT" in config["net_print"]` is the runtime switch that decides ViT vs CNN construction and pretrained loading — keep `net_print` consistent if you add a backbone.
- mAP and PR are only computed on epochs where `epoch % config["test_map"] == 0`; the "best" checkpoint is gated on improving mAP and triggers a PR-curve dump to the results `.txt`.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (60-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk go test             # Go test failures only (90%)
rtk jest                # Jest failures only (99.5%)
rtk vitest              # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk pytest              # Python test failures only (90%)
rtk rake test           # Ruby test failures only (90%)
rtk rspec               # RSpec test failures only (60%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%)
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->