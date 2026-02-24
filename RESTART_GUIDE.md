# Nanochat Resume Guide

Last updated: 2026-02-24 19:12 +08:00
Branch: `yxb/moe-add-sft-claude`

## Current State

- Training is currently paused.
- 8 GPUs are idle (`0% util`, `0 MiB` used on all devices at 19:11 +08:00).
- Latest consolidated docs:
  - `tmp_report/20260224_overall_progress_summary.md`
  - `tmp_report/20260224_1821_capacity_resume_run4_done.md`
  - `tmp_report/20260224_1829_capacity_opt_bufreuse_vs_cap1p0.md`

## Completed Work (Do Not Re-run)

### Capacity sweep baseline + remaining legs

Baseline experiment directory:
- `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_172715_moe-fp8-capacity-ablation-v1`

Resumed full four-leg experiment directory:
- `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_175412_moe-fp8-capacity-ablation-v1-resume`

Final metrics snapshot:

| run | tail50 tok/sec | tail50 dt (ms) | tail50 mfu | peak mem (MiB) | total time (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_cap0p0_baseline | 468782.94 | 1120.38 | 5.6722 | 22790.06 | 5.21 |
| static_cap1p0 | 785203.42 | 670.31 | 9.5014 | 18983.58 | 3.00 |
| static_cap1p25 | 747298.44 | 706.31 | 9.0426 | 19722.27 | 3.13 |
| dynamic_cap1p0 | 749886.60 | 701.27 | 9.0742 | 19057.08 | 3.15 |
| nocompile_cap1p0 | 363122.12 | 1449.95 | 4.3932 | 22808.69 | 6.34 |

Decision:
- Keep fixed-capacity `--moe-capacity-factor 1.0` as default for FP8+MoE static compile path.

### Small-step optimization iteration completed

Optimization experiment directory:
- `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_182427_moe-fp8-capacity-opt-v2-buffer`

Code change:
- `nanochat/moe.py` fixed-capacity dispatch path now reuses per-forward padded buffers to reduce allocation churn.

Validation:
- `source .venv/bin/activate && PYTHONPATH=. pytest tests/test_moe.py -q` (6 passed)

Result vs previous static `cap1.0`:

| metric | previous static_cap1p0 | static_cap1p0_bufreuse | delta |
| --- | ---: | ---: | ---: |
| tail50 tok/sec | 785203.42 | 806400.06 | +2.70% |
| tail50 dt (ms) | 670.31 | 650.67 | -2.93% |
| tail50 mfu | 9.5014 | 9.7588 | +2.71% |
| peak mem (MiB) | 18983.58 | 18987.83 | +0.02% |
| total time (min) | 3.00 | 2.97 | -1.00% |

## Resume Environment

```bash
cd /home/yangxiaobo/my_tools/books/nanochat
git checkout yxb/moe-add-sft-claude
git status --short --branch
source .venv/bin/activate
```

## Remaining Work (Next Codex Should Continue Here)

Goal: start next small hypothesis after buffer-reuse v2; do not rerun completed four legs unless explicitly requested.

Suggested next hypothesis:
- reduce fixed-capacity dispatch overhead further by minimizing per-step indexing/sorting overhead in slot construction while preserving numerical behavior.

Execution template for next experiment (new fresh dir, never reuse old dir):

```bash
source .venv/bin/activate
EXP_DIR="/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/$(date +%Y%m%d_%H%M%S)_moe-fp8-capacity-opt-v3-<tag>"
mkdir -p "$EXP_DIR/ckpt_root"

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --run dummy \
  --use-moe \
  --n-experts 8 \
  --expert-topk 2 \
  --moe-freq 1 \
  --expert-hidden-mult 2 \
  --balance-loss-coeff 0.01 \
  --depth 12 \
  --aspect-ratio 64 \
  --head-dim 64 \
  --max-seq-len 1024 \
  --device-batch-size 16 \
  --num-iterations 280 \
  --eval-every -1 \
  --core-metric-every -1 \
  --sample-every -1 \
  --save-every 280 \
  --checkpoint-root "$EXP_DIR/ckpt_root" \
  --fp8 \
  --fp8-include-moe-experts \
  --model-tag moe_fp8_experts_static_cap1p0_v3 \
  --compile-mode static \
  --moe-capacity-factor 1.0 2>&1 | tee "$EXP_DIR/moe_fp8_experts_static_cap1p0_v3.log"
```

Then regenerate report immediately:

```bash
python tmp_report/generate_moe_fp8_ablation_report.py \
  --exp-dir "$EXP_DIR" \
  --out-md "tmp_report/$(date +%Y%m%d_%H%M)_capacity_opt_v3_progress.md"
```

## Safety/Path Constraints Reminder

- Main branch must remain `yxb/moe-add-sft-claude`.
- Only write experiment outputs/checkpoints under:
  `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/`
- Treat `/mnt/stepeval/yangxiaobo/cache/nanochat` (except `tmp_exp/`) as read-only.
- Preserve unrelated dirty/untracked files.
