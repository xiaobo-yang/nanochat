# Compile Sweep v1 Snapshot (Run1 Done, Run2 Live)

- Snapshot time: 2026-02-24 17:01 HKT
- Branch: `yxb/moe-add-sft-claude`
- Experiment root: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`
- Queue session: `15105`

## Queue State

From `launch.txt`:

- `2026-02-24T16:54:03+08:00` start `moe_bf16_experts_static_long`
- `2026-02-24T16:58:21+08:00` finish `moe_bf16_experts_static_long` status=0
- `2026-02-24T16:58:22+08:00` start `moe_fp8_noexperts_static_long`

## Run1 Final Metrics

From `tmp_report/metrics/moe_fp8_ablation_summary.csv`:

- Run: `moe_bf16_experts_static_long`
- Status: `completed`
- Steps: `360`
- Last loss: `3.521940`
- tail50 tok/sec: `1,080,430.24`
- tail50 bf16_mfu: `13.0732`
- tail50 dt_ms: `487.11`
- Peak memory: `23,428.52 MiB`
- Total training time: `3.31m`

## Run2 Live Progress

From live session output (`moe_fp8_noexperts_static_long`):

- Latest observed step: `00026/00360 (7.22%)`
- Live loss: `6.126665`
- Live tok/sec: `504,662`
- Live bf16_mfu: `6.11`
- Live dt: `1038.89ms`

## Related Artifacts

- `tmp_report/20260224_1700_compile_sweep_v1_progress.md`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
