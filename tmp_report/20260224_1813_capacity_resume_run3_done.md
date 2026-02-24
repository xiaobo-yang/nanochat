# Capacity Resume Progress (Run3 Complete)

Generated at: 2026-02-24 18:13:30 +08:00

- Experiment: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_175412_moe-fp8-capacity-ablation-v1-resume`
- Newly completed leg: `moe_fp8_experts_dynamic_cap1p0`
- Remaining leg: `moe_fp8_experts_nocompile_cap1p0`

## Key metrics (tail50 window)

| run | tail50 tok/sec | tail50 dt (ms) | tail50 mfu | peak mem (MiB) | total time (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| moe_fp8_experts_static_cap1p0 | 785,203.42 | 670.31 | 9.5014 | 18,983.58 | 3.00 |
| moe_fp8_experts_dynamic_cap1p0 | 749,886.60 | 701.27 | 9.0742 | 19,057.08 | 3.15 |

## Notes

- `dynamic + cap1.0` did not beat `static + cap1.0` on throughput or latency in this 280-step sweep.
- Peak memory and total wall time are also slightly worse than `static + cap1.0`.
- Next step is finishing `nocompile + cap1.0` for full four-leg comparison.
