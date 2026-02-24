# Queue Recovery Snapshot (Run4 Done, Run5 Relaunched)

- Snapshot time: 2026-02-24 16:51 HKT
- Branch: `yxb/moe-add-sft-claude`
- Experiment root: `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_154946_moe-fp8-compile-ablation-v4`

## Run4 Final Metrics

From `tmp_report/metrics/moe_fp8_ablation_summary.csv`:

- Run: `moe_fp8_experts_nocompile_long`
- Status: `completed`
- Steps: `360`
- Last loss: `3.531636`
- tail50 tok/sec: `366,418.66`
- tail50 bf16_mfu: `4.4342`
- tail50 dt_ms: `1434.67`
- Peak memory: `22,867.05 MiB`
- Total training time: `8.46m`

## Queue Interruption + Recovery

From `launch.txt` and live session logs:

- Queue finished `moe_fp8_experts_nocompile_long` at `2026-02-24T16:48:59+08:00`
- A shell parse error occurred before the scripted handoff to run5, so queue runner exited early
- Recovery action: manually relaunched run5 on all 8 GPUs at `2026-02-24T16:49:54+08:00`
- Active session: `65097`
- Relaunched run: `moe_bf16_experts_nocompile_long`

## Run5 Live Status

From `moe_bf16_experts_nocompile_long.log` tail at snapshot:

- Latest observed step: `00043/00360 (11.94%)`
- Live loss: `5.778309`
- Live tok/sec: `768,246`
- Live bf16_mfu: `9.30`
- Live dt: `682.45ms`
- ETA: `5.9m`

## GPU Utilization Sample

`nvidia-smi` at snapshot:

- GPU0 100% (26.4 GiB)
- GPU1 53% (26.4 GiB)
- GPU2 92% (26.4 GiB)
- GPU3 41% (26.4 GiB)
- GPU4 81% (26.4 GiB)
- GPU5 48% (26.4 GiB)
- GPU6 30% (26.4 GiB)
- GPU7 71% (26.4 GiB)

## Notes

- The interruption happened during active script edits while the queue script process was already running.
- Follow-up batch launches will use the now-committed queue script revisions and avoid in-place edits on the currently executing script file.
