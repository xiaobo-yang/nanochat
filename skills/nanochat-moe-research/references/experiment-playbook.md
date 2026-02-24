# Experiment Playbook

## Purpose
Use this playbook to run reproducible MoE/SFT experiments in nanochat with strict branch, path, and commit discipline.

## Run Directory Convention
1. Create a dedicated run directory under:
   `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/`
2. Include:
   - `command.txt`
   - `notes.md`
   - `metrics.md` or `metrics.json`
   - optional `stdout.log`

## Command Patterns

### Baseline smoke run
```bash
python -m scripts.base_train \
  --run dummy \
  --depth 4 \
  --max-seq-len 512 \
  --device-batch-size 1 \
  --eval-tokens 512 \
  --core-metric-every -1 \
  --total-batch-size 512 \
  --num-iterations 20
```

### 8-GPU MoE run
```bash
torchrun --nproc_per_node=8 -m scripts.base_train \
  --run <run-name> \
  --use-moe \
  --n-experts 8 \
  --expert-topk 2 \
  --moe-freq 1 \
  --expert-hidden-mult 2 \
  --balance-loss-coeff 0.01 \
  --model-tag <model-tag>
```

### 8-GPU eval run
```bash
torchrun --nproc_per_node=8 -m scripts.base_eval \
  --model <model-tag>
```

## Notes Template
```markdown
# Hypothesis

# Change

# Validation Commands

# Results

# Decision (keep/revert/follow-up)
```

## Commit Message Template
```text
<type>: <short title>

Context:
- ...

Changes:
- ...

Validation:
- <command>
- <key output>

Results:
- ...

Risks / Next:
- ...
```
