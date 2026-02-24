# Capacity Sweep Live Snapshot (run1)

- Timestamp: 2026-02-24 17:33 +08:00
- Experiment: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_172715_moe-fp8-capacity-ablation-v1`
- Active run: `moe_fp8_experts_static_cap0p0`
- Queue launcher: `runs/moe_fp8_capacity_ablation_queue.sh`

## Current Progress

- Steps observed: `90`
- Last step: `89/280`
- Last loss: `5.112689`
- tail50 tok/sec: `434,792`
- tail50 dt: `1207.88 ms`
- tail50 bf16_mfu: `5.2616`

## Runtime Notes

- 8 GPUs are occupied (`nproc_per_node=8`) and have stable utilization.
- Static compile baseline (`capacity_factor=0.0`) still hits `torch._dynamo recompile_limit` in `nanochat/moe.py:14`.
- This run serves as baseline for comparing whether fixed capacity (`1.0`, `1.25`) reduces recompiles and improves throughput.

## Next Checkpoints

1. Wait for run1 completion and collect full tail metrics.
2. Verify queue transition to `moe_fp8_experts_static_cap1p0` with no GPU idle gap.
3. Regenerate report and compare `tail50 tok/sec` and `tail50 dt` against run1 baseline.
