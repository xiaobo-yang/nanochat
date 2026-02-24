# MoE FP8 Compile Ablation Progress

Generated at: 2026-02-24 17:35:38

Experiment dir: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_172715_moe-fp8-capacity-ablation-v1`

## Run Summary

| run | status | steps_observed | last_step | last_loss | tail50_tok/sec | tail50_mfu | tail50_dt_ms | peak_mem_mib | total_time_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moe_fp8_experts_static_cap0p0 | completed | 280 | 279 | 3.7182 | 468783 | 5.67 | 1120.4 | 22790.06 | 5.21 |
| moe_fp8_experts_static_cap1p0 | running | 7 | 6 | 8.0234 | 660110 | 7.99 | 2513.4 | - | - |

## Dynamic Compile Status

- No dynamic compile runs found in this experiment directory.

## Artifacts

- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
