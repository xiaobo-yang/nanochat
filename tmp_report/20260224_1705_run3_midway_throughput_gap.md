# Compile Sweep v1 Snapshot (Run3 Midway)

- Snapshot time: 2026-02-24 17:05 HKT
- Branch: `yxb/moe-add-sft-claude`
- Experiment root: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`
- Queue session: `15105`

## Queue State

- `moe_bf16_experts_static_long`: completed
- `moe_fp8_noexperts_static_long`: completed
- `moe_fp8_experts_static_long`: running

From `launch.txt` latest events:

- `2026-02-24T17:02:42+08:00` finish `moe_fp8_noexperts_static_long` status=0
- `2026-02-24T17:02:43+08:00` start `moe_fp8_experts_static_long`

## Run3 Live Metrics

From `tmp_report/metrics/moe_fp8_ablation_summary.csv`:

- Run: `moe_fp8_experts_static_long`
- Status: `running`
- Steps observed: `114`
- Latest step: `00113/00360`
- Last loss: `4.774159`
- tail50 tok/sec: `456,216.12`
- tail50 bf16_mfu: `5.5206`
- tail50 dt_ms: `1151.50`

## Resource Snapshot

From `nvidia-smi` during run3:

- GPU util across 8 cards: `80%` to `94%`
- Memory usage per card: around `26.5 GiB`

## Performance Delta vs Run2

Reference run2 (`moe_fp8_noexperts_static_long`) final tail50:

- tok/sec: `1,039,883.30`
- bf16_mfu: `12.5836`
- dt_ms: `505.19`

Current run3 tail50 vs run2 shows:

- `~56.1%` lower tok/sec
- `~56.1%` lower bf16_mfu
- `~127.9%` higher step dt

Interpretation:

- FP8 + static compile with expert kernels included is still heavily shape-unstable for MoE routing workloads in this configuration.
- The compile/recompile overhead is likely dominating throughput and blocking expected speedups.

## Related Artifacts

- `tmp_report/20260224_1705_compile_sweep_v1_progress.md`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
