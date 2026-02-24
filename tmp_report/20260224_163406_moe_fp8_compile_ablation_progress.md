# MoE FP8 Compile Ablation Progress

Generated at: 2026-02-24 16:34:07

Experiment dir: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_154946_moe-fp8-compile-ablation-v4`

## Run Summary

| run | status | steps_observed | last_step | last_loss | tail50_tok/sec | tail50_mfu | tail50_dt_ms | peak_mem_mib | total_time_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moe_fp8_noexperts_static | completed | 140 | 139 | 4.6716 | 982070 | 11.88 | 537.8 | 23231.91 | 1.59 |
| moe_fp8_experts_static | completed | 140 | 139 | 4.6735 | 475081 | 5.75 | 1104.4 | 22790.56 | 2.54 |
| moe_fp8_experts_dynamic | failed | 1 | 0 | 10.3978 | 12912 | 0.16 | 40602.2 | - | - |
| moe_fp8_experts_dynamic_coalesce0 | failed | 1 | 0 | 10.3978 | 30071 | 0.36 | 17434.8 | - | - |
| moe_fp8_experts_nocompile | completed | 140 | 139 | 4.6738 | 367435 | 4.45 | 1427.6 | 22867.05 | 3.23 |
| moe_fp8_experts_static_repeat | completed | 140 | 139 | 4.6742 | 478362 | 5.79 | 1097.0 | 22787.25 | 2.54 |
| moe_fp8_experts_nocompile_repeat2 | completed | 140 | 139 | 4.6729 | 364789 | 4.41 | 1440.6 | 22864.65 | 3.23 |
| moe_bf16_experts_dynamic | completed | 140 | 139 | 4.6605 | 947987 | 11.47 | 556.6 | 21107.20 | 1.71 |
| moe_fp8_dynamic_noexpertfp8 | completed | 140 | 139 | 4.6739 | 913356 | 11.05 | 579.6 | 20979.88 | 1.73 |
| moe_fp8_dynamic_noexpertfp8_long | running | 251 | 250 | 3.6994 | 934082 | 11.30 | 565.3 | - | - |
| moe_fp8_experts_dynamic_coalesce0_retry2 | completed | 140 | 139 | 4.6724 | 781338 | 9.45 | 673.0 | 19669.95 | 1.64 |
| moe_fp8_experts_dynamic_long | completed | 360 | 359 | 3.5310 | 799038 | 9.67 | 657.4 | 19665.65 | 4.08 |
| moe_fp8_experts_static.fail1 | failed | 1 | 0 | 10.3978 | 40703 | 0.49 | 12880.8 | - | - |

## Dynamic Compile Status

- `moe_fp8_experts_dynamic` failed with: `InductorError: AssertionError: -859663773703483/1000000000000000`
- `moe_fp8_experts_dynamic_coalesce0` failed with: `InductorError: AssertionError: -168373925552093/200000000000000`
- `moe_bf16_experts_dynamic` status: `completed`
- `moe_fp8_dynamic_noexpertfp8` status: `completed`
- `moe_fp8_dynamic_noexpertfp8_long` status: `running`
- `moe_fp8_experts_dynamic_coalesce0_retry2` status: `completed`
- `moe_fp8_experts_dynamic_long` status: `completed`

## Artifacts

- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
