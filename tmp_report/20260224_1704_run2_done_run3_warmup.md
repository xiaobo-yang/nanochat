# Compile Sweep v1 Snapshot (Run2 Done, Run3 Warmup)

- Snapshot time: 2026-02-24 17:04 HKT
- Branch: `yxb/moe-add-sft-claude`
- Experiment root: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`
- Queue session: `15105`

## Queue State

From `launch.txt`:

- `2026-02-24T16:54:03+08:00` start `moe_bf16_experts_static_long`
- `2026-02-24T16:58:21+08:00` finish `moe_bf16_experts_static_long` status=0
- `2026-02-24T16:58:22+08:00` start `moe_fp8_noexperts_static_long`
- `2026-02-24T17:02:42+08:00` finish `moe_fp8_noexperts_static_long` status=0
- `2026-02-24T17:02:43+08:00` start `moe_fp8_experts_static_long`

## Run2 Final Metrics

From `tmp_report/metrics/moe_fp8_ablation_summary.csv`:

- Run: `moe_fp8_noexperts_static_long`
- Status: `completed`
- Steps: `360`
- Last loss: `3.530226`
- tail50 tok/sec: `1,039,883.30`
- tail50 bf16_mfu: `12.5836`
- tail50 dt_ms: `505.19`
- Peak memory: `23,234.33 MiB`
- Total training time: `3.50m`

## Run3 Live Warmup

From live log `moe_fp8_experts_static_long.log`:

- Latest observed step: `00040/00360 (11.11%)`
- Live loss: `5.828435`
- tail40 tok/sec (warmup): `351,057`
- tail40 bf16_mfu (warmup): `4.25`
- tail40 dt_ms (warmup): `1761.51`

## Notable Runtime Signals

- `torch._dynamo hit config.recompile_limit (8)` warnings are repeatedly observed on all ranks.
- Recompile reason includes dynamic token routing shape mismatch in `nanochat/moe.py:12` (`tensor 'x' size mismatch at index 0`).
- This likely explains the major throughput gap versus `moe_fp8_noexperts_static_long` during early/mid warmup.

## Related Artifacts

- `tmp_report/20260224_1704_compile_sweep_v1_progress.md`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
