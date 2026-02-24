# Nanochat MoE Research Progress Summary (Paused)

- Generated at: 2026-02-24 17:40 +08:00
- Repo: `/home/yangxiaobo/my_tools/books/nanochat`
- Branch: `yxb/moe-add-sft-claude`
- Status: **Paused by user request**

## Current Runtime Status

- Active queue/training has been stopped.
- Verified no active `torchrun`/queue process remains.
- Verified all 8 GPUs are idle (`0% util`, `0 MiB` used).

## Main Code/Infra Progress

### 1) Workflow constraints and local agent tooling

- Added repository-level instructions file: `AGENTS.md`
- Added local research skill package:
  - `skills/nanochat-moe-research/SKILL.md`
  - `skills/nanochat-moe-research/references/experiment-playbook.md`
  - `skills/nanochat-moe-research/scripts/new_exp.py`
  - `skills/nanochat-moe-research/agents/openai.yaml`

### 2) MoE capacity-factor feature (core change)

- Added fixed-capacity MoE dispatch support via `--moe-capacity-factor`
- Plumbed config through training/model stack:
  - `scripts/base_train.py`
  - `nanochat/gpt.py`
  - `nanochat/moe.py`
- Added tests for capacity-factor behavior:
  - `tests/test_moe.py`

### 3) Experiment automation and reporting

- Added/updated queue scripts for 8-GPU ablations:
  - `runs/moe_fp8_capacity_ablation_queue.sh`
- Added/used report generation script:
  - `tmp_report/generate_moe_fp8_ablation_report.py`
- Maintained periodic markdown snapshots and figure/csv artifacts under `tmp_report/`

## Experiment Progress

## A. Compile Sweep V1 (Completed)

- Experiment dir:
  `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`
- Launch log shows queue completed successfully.

| run | status | steps | tail50 tok/sec | tail50 dt(ms) | tail50 mfu | peak mem (MiB) | total time (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `moe_bf16_experts_static_long` | completed | 360 | 1,080,430.24 | 487.11 | 13.0732 | 23,428.52 | 3.31 |
| `moe_fp8_noexperts_static_long` | completed | 360 | 1,039,883.30 | 505.19 | 12.5836 | 23,234.33 | 3.50 |
| `moe_fp8_experts_dynamic_long_retest` | completed | 360 | 785,495.06 | 669.74 | 9.5050 | 19,668.76 | 4.06 |
| `moe_fp8_experts_static_long` | completed | 360 | 474,178.48 | 1107.05 | 5.7366 | 22,790.91 | 6.67 |
| `moe_bf16_experts_nocompile_long_retest` | completed | 360 | 1,045,350.76 | 503.93 | 12.6498 | 23,433.47 | 3.41 |
| `moe_fp8_experts_nocompile_long_retest` | completed | 360 | 359,711.14 | 1462.56 | 4.3524 | 22,862.51 | 8.53 |

## B. Capacity Ablation V1 (Partially Completed, then manually paused)

- Experiment dir:
  `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_172715_moe-fp8-capacity-ablation-v1`
- Queue progression before pause:
  - `17:29:03` start `moe_fp8_experts_static_cap0p0`
  - `17:35:07` finish `moe_fp8_experts_static_cap0p0` status=0
  - `17:35:08` start `moe_fp8_experts_static_cap1p0`
  - `17:36:49` finish `moe_fp8_experts_static_cap1p0` status=1
  - `17:36:50` start `moe_fp8_experts_static_cap1p25`
- Pause action sent `SIGINT` to queue/torchrun, so run2/run3 are incomplete by design.

| run | status at pause | steps observed | last step | tail50 tok/sec | tail50 dt(ms) | tail50 mfu | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `moe_fp8_experts_static_cap0p0` | completed | 280 | 279/280 | 468,782.94 | 1120.38 | 5.6722 | completed normally |
| `moe_fp8_experts_static_cap1p0` | interrupted | 107 | 106/280 | 783,041.76 | 670.11 | 9.4742 | interrupted by user-requested stop (KeyboardInterrupt) |
| `moe_fp8_experts_static_cap1p25` | not started effectively | 0 | - | - | - | - | empty log due immediate pause after queue handoff |

## Key Findings So Far

1. FP8 + experts under static compile without fixed capacity (`cap0.0`) remains much slower than BF16/static and FP8/no-experts baselines.
2. Introducing fixed capacity (`cap1.0`) showed a large throughput lift in the partial run (from ~469k to ~783k tok/sec tail window), suggesting capacity stabilization may reduce compile/recompile pathologies.
3. Capacity-ablation queue needs rerun from `cap1.0` onward to obtain complete apples-to-apples final metrics (total_time, peak_mem, stable tail windows).

## Recent Commits (Top)

- `3636a35` report: capture cap0.0 completion and cap1.0 warm-start transition
- `331ce28` report: snapshot live metrics for capacity sweep run1 (cap0.0 static fp8)
- `99b6b55` report: capture run5 completion and run6 live ramp in compile_sweep_v1
- `44c94a9` moe: add fixed-capacity dispatch mode and capacity ablation queue
- `287b302` report: log run4 completion and run5 nocompile live regression
- `a9d3934` report: record run3 completion and run4 dynamic throughput recovery

## Important Artifact Paths

- Compile sweep v1:
  `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_164636_moe-fp8-compile-sweep-v1`
- Capacity ablation v1:
  `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_172715_moe-fp8-capacity-ablation-v1`
- Ongoing report outputs:
  `tmp_report/`

## Resume Checklist (when restarting work)

1. Resume from branch `yxb/moe-add-sft-claude`.
2. Relaunch capacity ablation from a fresh directory under `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/`.
3. Re-run `cap1.0`, `cap1.25`, `dynamic cap1.0`, `nocompile cap1.0` to completion.
4. Regenerate summary plots and compare against `cap0.0` and compile_sweep_v1 baselines.

Detailed run commands and continuation procedure are documented in:
- `RESTART_GUIDE.md`
