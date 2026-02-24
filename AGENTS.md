# AGENTS.md

## Scope
- Work inside `/data/my_tools/books/nanochat`.
- Prioritize research and engineering iteration around MoE/SFT training, evaluation, and tooling in this codebase.

## Hard Constraints
- Keep the main working branch as `yxb/moe-add-sft-claude`.
- If a temporary branch is required, name it `yxb/moe-add-sft-claude/sub-<topic>`, merge back to main immediately after finishing, then delete the temporary branch.
- Do not delete any file outside the repository-local `nanochat/` directory.
- Treat `/mnt/stepeval/yangxiaobo/cache/nanochat` as read-only, except:
  `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/` is writable for experiment outputs and checkpoints.
- Save all experiment checkpoints only under:
  `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/`.
- Preserve existing unrelated or user-created dirty/untracked files unless explicitly asked to change them.

## Research Loop
1. Form one concrete hypothesis from current bottlenecks or quality gaps.
2. Make the smallest useful code/test change to validate the hypothesis.
3. Run targeted verification first (`pytest`, unit/smoke checks).
4. Run training/eval experiments (prefer 8-GPU `torchrun` when relevant).
5. Record commands, config, and outcomes in run artifacts under `tmp_exp`.
6. Commit with a detailed message before moving to the next hypothesis.

## Experiment Artifacts
- Create a unique run directory per experiment under `tmp_exp` (timestamp + short tag).
- Keep at least:
  - `command.txt`: exact command line.
  - `notes.md`: hypothesis, expected result, observed result.
  - `metrics.json` or `metrics.md`: key numbers and checkpoints.

## Commit Discipline
- Commit frequently at milestone boundaries:
  - docs/skills baseline
  - each meaningful code change
  - experiment configuration/result updates
- Use detailed commit messages that include:
  - problem or hypothesis
  - changed files and why
  - validation commands
  - observed results or limitations

## Safety
- Avoid destructive git operations (`reset --hard`, checkout reverts) unless explicitly requested.
- Never revert or overwrite changes that were not made in the current task.
