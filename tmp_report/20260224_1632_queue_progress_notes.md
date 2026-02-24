# MoE FP8 Queue Progress Notes (2026-02-24 16:32)

## Queue Status
- Active experiment root: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_154946_moe-fp8-compile-ablation-v4`
- Active queue tag: `20260224_162523`
- Completed in this queue:
  - `moe_fp8_experts_dynamic_long` (360 steps)
- Running now:
  - `moe_fp8_dynamic_noexpertfp8_long` (360 steps, started at 2026-02-24 16:30:21 +08:00)

## Fresh Result (Long Run)
| run | steps | tail50 tok/sec | tail50 mfu | tail50 dt ms | peak mem MiB | total time min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `moe_fp8_experts_dynamic_long` | 360 | 799038 | 9.67 | 657.40 | 19665.65 | 4.08 |

## Current Comparison Snapshot
| run | status | tail50 tok/sec | peak mem MiB | total time min |
| --- | --- | ---: | ---: | ---: |
| `moe_fp8_experts_static_repeat` | completed | 478362 | 22787.25 | 2.54 |
| `moe_fp8_experts_nocompile_repeat2` | completed | 364789 | 22864.65 | 3.23 |
| `moe_fp8_experts_dynamic_coalesce0_retry2` | completed | 781338 | 19669.95 | 1.64 |
| `moe_fp8_experts_dynamic_long` | completed | 799038 | 19665.65 | 4.08 |
| `moe_bf16_experts_dynamic` | completed | 947987 | 21107.20 | 1.71 |
| `moe_fp8_dynamic_noexpertfp8` | completed | 913356 | 20979.88 | 1.73 |

## Interim Takeaways
- Dynamic compile + stability guard (`coalesce_tiling_analysis=0`) stays stable for long FP8-with-experts runs.
- FP8-with-experts dynamic remains clearly faster than static and nocompile FP8-with-experts baselines.
- Peak memory for FP8-with-experts dynamic (`~19.67 GiB`) is substantially below static/nocompile FP8-with-experts (`~22.8 GiB`).

## Artifacts
- Latest auto report: `tmp_report/20260224_163021_moe_fp8_compile_ablation_progress.md`
- Auto CSV: `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- Auto figures:
  - `tmp_report/figures/moe_fp8_ablation_loss.png`
  - `tmp_report/figures/moe_fp8_ablation_toksec.png`
  - `tmp_report/figures/moe_fp8_ablation_mfu.png`
  - `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
