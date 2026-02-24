# MoE FP8 Compile Ablation Progress

Generated at: 2026-02-24 16:58:58

Experiment dir: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`

## Run Summary

| run | status | steps_observed | last_step | last_loss | tail50_tok/sec | tail50_mfu | tail50_dt_ms | peak_mem_mib | total_time_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moe_bf16_experts_static_long | completed | 360 | 359 | 3.5219 | 1080430 | 13.07 | 487.1 | 23428.52 | 3.31 |
| moe_fp8_noexperts_static_long | running | 7 | 6 | 8.0074 | 243418 | 2.95 | 3108.7 | - | - |

## Dynamic Compile Status

- No dynamic compile runs found in this experiment directory.

## Artifacts

- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
