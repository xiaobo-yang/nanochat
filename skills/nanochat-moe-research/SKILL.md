---
name: nanochat-moe-research
description: Structured workflow for nanochat MoE/SFT research. Use when asked to explore this codebase, implement MoE or SFT improvements, run controlled training/evaluation (including 8-GPU torchrun), enforce branch/path constraints, store checkpoints under /mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/, and produce detailed periodic commits.
---

# Nanochat Moe Research

## Workflow

1. Confirm branch and workspace safety.
   - Stay on `yxb/moe-add-sft-claude`.
   - If branching is necessary, use `yxb/moe-add-sft-claude/sub-<topic>`, merge back, then delete it.
   - Preserve existing user changes and untracked files.
2. Bound the research question.
   - Choose one concrete hypothesis (quality, stability, or throughput).
   - Identify the minimal files to modify and a measurable success criterion.
3. Implement minimally.
   - Prefer small diffs with explicit invariants.
   - Add or update tests near the modified logic.
4. Validate locally before long runs.
   - Run targeted unit/smoke checks first.
   - Run full/8-GPU experiments only after sanity checks pass.
5. Persist experiment artifacts.
   - Store checkpoints and run artifacts only under:
     `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/`.
   - Keep command, notes, and metrics in each run directory.
6. Commit at each milestone.
   - Use detailed commit messages with hypothesis, change scope, validation commands, and outcomes.

## Execution Checklist

- Run branch/status checks:
  - `git branch --show-current`
  - `git status --short --branch`
- Create experiment directory (use bundled script):
  - `python skills/nanochat-moe-research/scripts/new_exp.py --name <tag>`
- Run an 8-GPU training/eval command only when needed and reproducible:
  - `torchrun --nproc_per_node=8 -m scripts.base_train ...`
  - `torchrun --nproc_per_node=8 -m scripts.base_eval ...`
- Capture command and outputs inside the created experiment directory.

## Constraints

- Do not delete files outside repository-local `nanochat/`.
- Do not write outside `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/` for experiment artifacts.
- Treat `/mnt/stepeval/yangxiaobo/cache/nanochat` (excluding `tmp_exp`) as read-only.

## Resources

- Read `references/experiment-playbook.md` for command patterns and reporting format.
- Use `scripts/new_exp.py` to create timestamped experiment directories with a metadata skeleton.
