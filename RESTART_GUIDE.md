# Nanochat Resume Guide

Last updated: 2026-02-24 17:45 +08:00
Branch: `yxb/moe-add-sft-claude`

## Current State

- Training is paused intentionally.
- 8 GPUs are idle.
- Latest consolidated summary:
  - `tmp_report/20260224_overall_progress_summary.md`
  - `tmp_report/20260224_1740_capacity_sweep_paused_progress.md`

## Completed vs Pending

### Completed
- FP8 compile ablation sweep v1:
  - `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`
- Capacity ablation run `cap0.0`:
  - `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_172715_moe-fp8-capacity-ablation-v1/moe_fp8_experts_static_cap0p0.log`

### Incomplete / Need Re-run
- `cap1.0` static FP8 experts (interrupted by manual pause)
- `cap1.25` static FP8 experts (not effectively started)
- `dynamic cap1.0` FP8 experts
- `nocompile cap1.0` FP8 experts

## Resume Environment

```bash
cd /home/yangxiaobo/my_tools/books/nanochat
source .venv/bin/activate
git checkout yxb/moe-add-sft-claude
git status --short --branch
```

## Recommended Resume Plan

1. Create a fresh experiment directory under `tmp_exp` (do not reuse interrupted run dir).
2. Run remaining ablation legs one by one with 8 GPUs.
3. Regenerate reports after each leg.
4. Commit report updates at each milestone.

## Suggested Commands (Remaining Legs Only)

Set a new experiment directory:

```bash
EXP_DIR="/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/$(date +%Y%m%d_%H%M%S)_moe-fp8-capacity-ablation-v1-resume"
mkdir -p "$EXP_DIR/ckpt_root"
```

Common args:

```bash
COMMON_ARGS=(
  --run dummy
  --use-moe
  --n-experts 8
  --expert-topk 2
  --moe-freq 1
  --expert-hidden-mult 2
  --balance-loss-coeff 0.01
  --depth 12
  --aspect-ratio 64
  --head-dim 64
  --max-seq-len 1024
  --device-batch-size 16
  --num-iterations 280
  --eval-every -1
  --core-metric-every -1
  --sample-every -1
  --save-every 280
  --checkpoint-root "$EXP_DIR/ckpt_root"
  --fp8
  --fp8-include-moe-experts
)
```

`cap1.0` static:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  "${COMMON_ARGS[@]}" \
  --model-tag moe_fp8_experts_static_cap1p0_resume \
  --compile-mode static \
  --moe-capacity-factor 1.0 2>&1 | tee "$EXP_DIR/moe_fp8_experts_static_cap1p0_resume.log"
```

`cap1.25` static:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  "${COMMON_ARGS[@]}" \
  --model-tag moe_fp8_experts_static_cap1p25_resume \
  --compile-mode static \
  --moe-capacity-factor 1.25 2>&1 | tee "$EXP_DIR/moe_fp8_experts_static_cap1p25_resume.log"
```

`cap1.0` dynamic:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  "${COMMON_ARGS[@]}" \
  --model-tag moe_fp8_experts_dynamic_cap1p0_resume \
  --compile-mode dynamic \
  --moe-capacity-factor 1.0 2>&1 | tee "$EXP_DIR/moe_fp8_experts_dynamic_cap1p0_resume.log"
```

`cap1.0` nocompile:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  "${COMMON_ARGS[@]}" \
  --model-tag moe_fp8_experts_nocompile_cap1p0_resume \
  --compile-mode none \
  --moe-capacity-factor 1.0 2>&1 | tee "$EXP_DIR/moe_fp8_experts_nocompile_cap1p0_resume.log"
```

Generate progress report:

```bash
python tmp_report/generate_moe_fp8_ablation_report.py \
  --exp-dir "$EXP_DIR" \
  --out-md "tmp_report/$(date +%Y%m%d_%H%M)_capacity_resume_progress.md"
```

## Decision Gate

Treat `moe_fp8_experts_static_cap0p0` from the earlier run as baseline and compare:
- tail50 `tok/sec`
- tail50 `dt_ms`
- tail50 `bf16_mfu`
- peak memory and total time

If `cap1.0` static keeps a stable gain vs `cap0.0` and does not regress memory sharply, keep fixed-capacity dispatch as default candidate for FP8+MoE static mode.
