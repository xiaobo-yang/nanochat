# MoE FP8 Compile Ablation Progress

Generated at: 2026-02-24 17:25:57

Experiment dir: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`

## Run Summary

| run | status | steps_observed | last_step | last_loss | tail50_tok/sec | tail50_mfu | tail50_dt_ms | peak_mem_mib | total_time_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moe_bf16_experts_nocompile_long_retest | running | 60 | 59 | 5.5133 | 601536 | 7.28 | 996.1 | - | - |
| moe_bf16_experts_static_long | completed | 360 | 359 | 3.5219 | 1080430 | 13.07 | 487.1 | 23428.52 | 3.31 |
| moe_fp8_experts_dynamic_long_retest | completed | 360 | 359 | 3.5322 | 785495 | 9.50 | 669.7 | 19668.76 | 4.06 |
| moe_fp8_experts_nocompile_long_retest | completed | 360 | 359 | 3.5318 | 359711 | 4.35 | 1462.6 | 22862.51 | 8.53 |
| moe_fp8_experts_static_long | completed | 360 | 359 | 3.5315 | 474178 | 5.74 | 1107.0 | 22790.91 | 6.67 |
| moe_fp8_noexperts_static_long | completed | 360 | 359 | 3.5302 | 1039883 | 12.58 | 505.2 | 23234.33 | 3.50 |

## Dynamic Compile Status

- `moe_fp8_experts_dynamic_long_retest` status: `completed`

## Artifacts

- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
