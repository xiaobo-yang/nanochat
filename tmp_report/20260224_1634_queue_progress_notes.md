# Queue Progress Notes (Live Snapshot)

- Snapshot time: 2026-02-24 16:35 (UTC+8)
- Branch: `yxb/moe-add-sft-claude`
- Experiment root: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_154946_moe-fp8-compile-ablation-v4`
- Queue script: `runs/moe_fp8_compile_ablation_queue.sh`

## GPU Occupancy Sample

`nvidia-smi` sample:

- GPU0: 79% util, 24063 MiB
- GPU1: 73% util, 24057 MiB
- GPU2: 80% util, 24059 MiB
- GPU3: 81% util, 24063 MiB
- GPU4: 71% util, 24061 MiB
- GPU5: 74% util, 24061 MiB
- GPU6: 60% util, 24051 MiB
- GPU7: 75% util, 24055 MiB

## Queue State

`launch.txt` shows:

- `moe_fp8_experts_dynamic_long` finished at 16:30:19 with `status=0`
- `moe_fp8_dynamic_noexpertfp8_long` started at 16:30:21 and is currently running

Current training line (from `moe_fp8_dynamic_noexpertfp8_long.log`):

- `step 00280/00360 (77.78%) | loss: 3.647090 | dt: 547.94ms | tok/sec: 956,830 | bf16_mfu: 11.58 | total time: 3.17m | eta: 0.9m`

## Completed vs Running Metrics

From `tmp_report/metrics/moe_fp8_ablation_summary.csv` after refresh:

- `moe_fp8_experts_dynamic_long`: completed, `tail50_tok/sec=799,038`, `tail50_mfu=9.67`, `tail50_dt_ms=657.40`, `peak_mem=19665.65 MiB`, `time=4.08m`
- `moe_fp8_dynamic_noexpertfp8_long`: running, observed `251` steps at report generation time, `tail50_tok/sec=934,082`, `tail50_mfu=11.30`, `tail50_dt_ms=565.26`

Interim finding:

- In this long-run stage, disabling expert FP8 under dynamic compile currently gives higher throughput than expert-FP8 dynamic run, while memory remains in the same broad band (~20-24 GiB depending on run).

## Artifacts Updated This Snapshot

- `tmp_report/20260224_163406_moe_fp8_compile_ablation_progress.md`
- `tmp_report/metrics/moe_fp8_ablation_summary.csv`
- `tmp_report/metrics/moe_fp8_ablation_steps.csv`
- `tmp_report/figures/moe_fp8_ablation_loss.png`
- `tmp_report/figures/moe_fp8_ablation_toksec.png`
- `tmp_report/figures/moe_fp8_ablation_mfu.png`
- `tmp_report/figures/moe_fp8_ablation_dt_ms.png`
