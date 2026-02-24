# Recovery Complete + Next 8-GPU Sweep Started

- Snapshot time: 2026-02-24 16:57 HKT
- Branch: `yxb/moe-add-sft-claude`
- Previous experiment: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_154946_moe-fp8-compile-ablation-v4`
- Current queue experiment: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`

## Run5 Finalized (manual recovery run)

Run: `moe_bf16_experts_nocompile_long`

- Status: `completed`
- Last step: `359/360` (360 steps observed)
- Last loss: `3.522287`
- tail50 tok/sec: `1,029,301.24`
- tail50 bf16_mfu: `12.4546`
- tail50 dt_ms: `512.33`
- Peak memory: `23,427.59 MiB`
- Total training time: `3.42m`

`launch.txt` bookkeeping has been patched with:

- `[manual-recovery 2026-02-24T16:53:52+08:00] finish moe_bf16_experts_nocompile_long status=0`

## Fresh Queue Started Immediately

Queue session: `15105`

- Started at: `2026-02-24T16:54:03+08:00`
- Active run: `moe_bf16_experts_static_long`
- Queue log: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1/launch.txt`

Latest observed live progress at snapshot:

- Step: `00049/00360 (13.61%)`
- Loss: `5.672396`
- tok/sec: `643,640`
- bf16_mfu: `7.79`
- dt: `814.57ms`

## Updated Report Artifacts

- `tmp_report/20260224_1656_moe_fp8_compile_ablation_progress.md`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
