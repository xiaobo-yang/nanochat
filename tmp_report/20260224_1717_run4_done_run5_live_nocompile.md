# compile_sweep_v1 milestone: run4 done, run5 live

- timestamp: 2026-02-24 17:17:01 +0800
- experiment: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`
- queue log: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1/launch.txt`

## Queue state

- `17:15:08 +0800`: finished `moe_fp8_experts_dynamic_long_retest`
- `17:15:10 +0800`: started `moe_fp8_experts_nocompile_long_retest`
- transition was automatic with no relaunch gap.

## GPU utilization snapshot

`nvidia-smi` snapshot during run5:

- gpu0: 87% / 26531 MiB
- gpu1: 86% / 26523 MiB
- gpu2: 73% / 26527 MiB
- gpu3: 84% / 26527 MiB
- gpu4: 87% / 26527 MiB
- gpu5: 81% / 26531 MiB
- gpu6: 82% / 26521 MiB
- gpu7: 93% / 26525 MiB

## Key metrics (from `tmp_report/metrics/moe_fp8_ablation_summary.csv`)

- `moe_fp8_experts_dynamic_long_retest`: completed, tail50 tok/s `785,495`, tail50 mfu `9.50`, tail50 dt `669.7ms`, total time `4.06m`.
- `moe_fp8_experts_nocompile_long_retest`: running, `77` steps observed, last_step `76`, tail50 tok/s `348,575`, tail50 mfu `4.22`, tail50 dt `1507.1ms`.

## Live log snippet notes for run5

- run5 step 27-76 band shows:
  - tok/sec roughly `332k-370k`
  - bf16_mfu roughly `4.02-4.48`
  - dt roughly `1414-1579ms`
- loss decreases smoothly (`~10.40 -> ~5.28` by step 76), no runtime errors observed.

## Interpretation

- Dynamic compile retest (run4) recovered a large portion of the expert-path regression seen in static compile.
- No-compile expert retest (run5) is clearly slower in early stage than dynamic compile tail speed; full-run tail comparison is pending after completion.
- Current priority is to keep run5 and run6 contiguous to finish the sweep and produce final comparison.
