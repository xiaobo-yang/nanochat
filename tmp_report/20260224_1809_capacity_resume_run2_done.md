# Capacity Resume Progress (Run2 Complete)

Generated at: 2026-02-24 18:09:00 +08:00

- Experiment: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_175412_moe-fp8-capacity-ablation-v1-resume`
- Completed legs:
  - `moe_fp8_experts_static_cap1p0`
  - `moe_fp8_experts_static_cap1p25`
- Remaining legs:
  - `moe_fp8_experts_dynamic_cap1p0`
  - `moe_fp8_experts_nocompile_cap1p0`

## Key metrics (tail50 window)

| run | tail50 tok/sec | tail50 dt (ms) | tail50 mfu | peak mem (MiB) | total time (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| moe_fp8_experts_static_cap1p0 | 785,203.42 | 670.31 | 9.5014 | 18,983.58 | 3.00 |
| moe_fp8_experts_static_cap1p25 | 747,298.44 | 706.31 | 9.0426 | 19,722.27 | 3.13 |

## Notes

- Under static compile, `cap1.25` is slower than `cap1.0` on both throughput and latency.
- `cap1.25` also increases peak memory by ~739 MiB and total time by ~0.13 min.
- Next step is to finish `dynamic + cap1.0` and `nocompile + cap1.0` for full four-leg comparison.
