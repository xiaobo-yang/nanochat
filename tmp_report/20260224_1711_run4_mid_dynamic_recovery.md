# Compile Sweep v1 Snapshot (Run4 Midway, Dynamic Recovery)

- Snapshot time: 2026-02-24 17:11 HKT
- Branch: `yxb/moe-add-sft-claude`
- Experiment root: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`
- Queue session: `15105`

## Current Queue Position

- `moe_fp8_experts_static_long`: completed
- `moe_fp8_experts_dynamic_long_retest`: running
- Remaining after run4: `moe_fp8_experts_nocompile_long_retest`, `moe_bf16_experts_nocompile_long_retest`

## Run4 Live Metrics

From `tmp_report/metrics/moe_fp8_ablation_summary.csv`:

- Run: `moe_fp8_experts_dynamic_long_retest`
- Status: `running`
- Steps observed: `103`
- Latest step: `00102/00360`
- Last loss: `4.937291`
- tail50 tok/sec: `736,401.84`
- tail50 bf16_mfu: `8.9118`
- tail50 dt_ms: `713.44`

## Comparative View

Reference tails:

- `moe_fp8_experts_static_long`: `474,178.48 tok/s`, `5.7366 mfu`, `1107.05 ms`
- `moe_fp8_experts_dynamic_long_retest` (current): `736,401.84 tok/s`, `8.9118 mfu`, `713.44 ms`
- `moe_fp8_noexperts_static_long`: `1,039,883.30 tok/s`, `12.5836 mfu`, `505.19 ms`

Current deltas:

- Dynamic run4 vs static-experts run3: `+55.3% tok/s`, `+55.4% mfu`, `-35.6% dt`
- Dynamic run4 vs no-experts static run2: still `~29.2%` lower tok/s

Interpretation:

- Dynamic compile substantially recovers the expert-path performance lost in static compile mode.
- Expert routing overhead remains significant versus no-experts baseline, but the compile-mode penalty is clearly reduced.

## Resource Snapshot

From `nvidia-smi` during run4:

- GPU util across 8 cards: `53%` to `68%`
- Memory usage per card: around `22.5 GiB`

## Related Artifacts

- `tmp_report/20260224_1711_compile_sweep_v1_progress.md`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
