# Queue Transition Notes (Run2 -> Run3)

- Snapshot time: 2026-02-24 16:36 (UTC+8)
- Branch: `yxb/moe-add-sft-claude`
- Experiment root: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_154946_moe-fp8-compile-ablation-v4`

## Queue Transition Confirmed

From `launch.txt`:

- `moe_fp8_dynamic_noexpertfp8_long` finished at `16:35:11` with `status=0`
- `moe_bf16_experts_dynamic_long` started at `16:35:13`

## Run2 Final Metrics

From `tmp_report/metrics/moe_fp8_ablation_summary.csv`:

- Run: `moe_fp8_dynamic_noexpertfp8_long`
- Status: `completed`
- Steps: `360`
- Last loss: `3.528897`
- tail50 tok/sec: `963503.82`
- tail50 bf16_mfu: `11.6596`
- tail50 dt_ms: `545.02`
- Peak memory: `20983.23 MiB`
- Total training time: `3.89m`

## Run3 Live Status

Current line from `moe_bf16_experts_dynamic_long.log`:

- `step 00024/00360 (6.67%) | loss: 6.174623 | dt: 1096.48ms | tok/sec: 478,154 | bf16_mfu: 5.79 | total time: 0.39m | eta: 9.3m`

GPU utilization sample at snapshot:

- GPU0 33%, GPU1 38%, GPU2 31%, GPU3 35%, GPU4 23%, GPU5 32%, GPU6 31%, GPU7 30%
- Memory residency already restored at ~24 GiB per device

## Interim Comparison

- `moe_fp8_dynamic_noexpertfp8_long` currently leads `moe_fp8_experts_dynamic_long` by ~20.6% in tail-50 throughput
- Dynamic long-run memory remains in the ~19.7-21.0 GiB range for the completed runs so far
