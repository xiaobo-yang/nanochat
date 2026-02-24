# MoE FP8 Compile Ablation Progress

Generated at: 2026-02-24 18:28:24

Experiment dir: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_182427_moe-fp8-capacity-opt-v2-buffer`

## Run Summary

| run | status | steps_observed | last_step | last_loss | tail50_tok/sec | tail50_mfu | tail50_dt_ms | peak_mem_mib | total_time_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moe_fp8_experts_static_cap1p0_bufreuse | completed | 280 | 279 | 3.7238 | 806400 | 9.76 | 650.7 | 18987.83 | 2.97 |

## Dynamic Compile Status

- No dynamic compile runs found in this experiment directory.

## Artifacts

- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
