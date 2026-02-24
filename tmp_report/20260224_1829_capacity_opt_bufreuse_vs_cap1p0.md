# Capacity Resume Final + Optimization Iteration 1

Generated at: 2026-02-24 18:29 +08:00

## Four-leg completion snapshot

| run | tail50 tok/sec | tail50 dt (ms) | tail50 mfu | peak mem (MiB) | total time (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_cap0p0_baseline | 468782.94 | 1120.38 | 5.6722 | 22790.06 | 5.21 |
| static_cap1p0 | 785203.42 | 670.31 | 9.5014 | 18983.58 | 3.00 |
| static_cap1p25 | 747298.44 | 706.31 | 9.0426 | 19722.27 | 3.13 |
| dynamic_cap1p0 | 749886.60 | 701.27 | 9.0742 | 19057.08 | 3.15 |
| nocompile_cap1p0 | 363122.12 | 1449.95 | 4.3932 | 22808.69 | 6.34 |

## Small-step optimization (buffer reuse in fixed-capacity path)

- hypothesis: avoid per-expert temporary allocation churn in `MoE._dispatch_fixed_capacity` by reusing per-forward padded buffers.
- run: `moe_fp8_experts_static_cap1p0_bufreuse` in `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_182427_moe-fp8-capacity-opt-v2-buffer`

| metric | previous static_cap1p0 | static_cap1p0_bufreuse | delta |
| --- | ---: | ---: | ---: |
| tail50 tok/sec | 785203.42 | 806400.06 | +2.70% |
| tail50 dt (ms) | 670.31 | 650.67 | -2.93% |
| tail50 mfu | 9.5014 | 9.7588 | +2.71% |
| peak mem (MiB) | 18983.58 | 18987.83 | +0.02% |
| total time (min) | 3.00 | 2.97 | -1.00% |

## Recommendation

- keep fixed-capacity (`moe_capacity_factor=1.0`) as default for FP8+MoE static compile path.
- reason: strong gain vs cap0.0 baseline and stable win over dynamic/nocompile variants, with much lower latency and shorter wall time.
- keep `cap1.25` as non-default tuning option; it underperforms cap1.0 on this workload.
