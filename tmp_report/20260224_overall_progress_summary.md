# Nanochat MoE Research Progress Summary

- Generated at: 2026-02-24 19:12 +08:00
- Repo: `/home/yangxiaobo/my_tools/books/nanochat`
- Branch: `yxb/moe-add-sft-claude`
- Status: Ready for next optimization iteration (no active run)

## Runtime Status

- No active `torchrun` or training queue process.
- 8 GPUs verified idle at 19:11 +08:00:
  - GPU0..GPU7: `0% util`, `0 MiB / 81559 MiB`.

## Completed Milestones

### 1) Capacity ablation remaining legs completed from fresh resume directory

Baseline directory:
- `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_172715_moe-fp8-capacity-ablation-v1`

Fresh resumed directory:
- `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_175412_moe-fp8-capacity-ablation-v1-resume`

Runs completed:
- `static + cap1.0`
- `static + cap1.25`
- `dynamic + cap1.0`
- `nocompile + cap1.0`

Final comparison (tail50 window):

| run | tail50 tok/sec | tail50 dt (ms) | tail50 mfu | peak mem (MiB) | total time (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_cap0p0_baseline | 468782.94 | 1120.38 | 5.6722 | 22790.06 | 5.21 |
| static_cap1p0 | 785203.42 | 670.31 | 9.5014 | 18983.58 | 3.00 |
| static_cap1p25 | 747298.44 | 706.31 | 9.0426 | 19722.27 | 3.13 |
| dynamic_cap1p0 | 749886.60 | 701.27 | 9.0742 | 19057.08 | 3.15 |
| nocompile_cap1p0 | 363122.12 | 1449.95 | 4.3932 | 22808.69 | 6.34 |

Conclusion:
- Fixed-capacity `cap1.0` static is the best default on this workload.
- `cap1.25` is worse than `cap1.0` on throughput/latency and uses more memory.
- `nocompile` remains clearly slower and higher-memory.

Primary report files:
- `tmp_report/20260224_1821_capacity_resume_run4_done.md`
- `tmp_report/20260224_1829_capacity_opt_bufreuse_vs_cap1p0.md`

### 2) Next-round small optimization implemented and verified

Hypothesis:
- reduce allocation churn in fixed-capacity dispatch by reusing padded per-forward buffers.

Code:
- `nanochat/moe.py` (fixed-capacity path buffer reuse)

Validation:
- `source .venv/bin/activate && PYTHONPATH=. pytest tests/test_moe.py -q` -> 6 passed

Experiment directory:
- `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_182427_moe-fp8-capacity-opt-v2-buffer`

Result (`moe_fp8_experts_static_cap1p0_bufreuse`):
- tail50 tok/sec: `806400.06`
- tail50 dt: `650.67 ms`
- tail50 mfu: `9.7588`
- peak mem: `18987.83 MiB`
- total time: `2.97 min`

Delta vs previous static `cap1.0`:
- tok/sec `+2.70%`
- dt `-2.93%`
- mfu `+2.71%`
- peak mem `+0.02%`
- total time `-1.00%`

Supporting report files:
- `tmp_report/20260224_1828_capacity_opt_bufreuse_done.md`
- `tmp_report/20260224_1829_capacity_opt_bufreuse_vs_cap1p0.md`

## Latest Relevant Commits

- `ba52023` report: add buffer-reuse optimization run and final capacity comparison snapshot
- `452d3f4` moe: reduce fixed-capacity dispatch allocations with per-forward padded buffers
- `4967ca2` report: finalize capacity resume run4 (nocompile cap1.0) and full four-leg sweep
- `06bbdd9` report: complete capacity resume run3 dynamic cap1.0
- `bc5bd1e` report: complete capacity resume run2 static cap1.25 and snapshot metrics
- `c2b6587` report: complete resumed capacity leg static+cap1.0 with fresh exp dir

## Resume Commands (for new Codex session)

```bash
cd /home/yangxiaobo/my_tools/books/nanochat
git checkout yxb/moe-add-sft-claude
git status --short --branch
source .venv/bin/activate
```

Before any `torchrun`, always run:

```bash
source .venv/bin/activate
```

Start next optimization run with a fresh experiment directory only:

```bash
source .venv/bin/activate
EXP_DIR="/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/$(date +%Y%m%d_%H%M%S)_moe-fp8-capacity-opt-v3-<tag>"
mkdir -p "$EXP_DIR/ckpt_root"
```

## Constraints Reminder

- Keep branch at `yxb/moe-add-sft-claude`.
- Write checkpoints/artifacts only under `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/`.
- Do not modify other paths under `/mnt/stepeval/yangxiaobo/cache/nanochat`.
- Preserve unrelated user files and dirty/untracked content.
