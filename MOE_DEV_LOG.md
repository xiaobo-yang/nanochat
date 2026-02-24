# MoE Development Log

## Architecture Changes

### Overview
Integrated Mixture-of-Experts (MoE) as an optional drop-in replacement for dense MLP layers in the GPT model. The implementation is fully backward-compatible: when `use_moe=False` (default), the model behaves identically to before.

### Files Modified

| File | Change |
|------|--------|
| `nanochat/moe.py` | Expert MLP uses `config.expert_hidden_mult` instead of hardcoded `4` |
| `nanochat/gpt.py` | MoE integration: config, Block, init_weights, forward, FLOPs estimation |
| `scripts/base_train.py` | CLI args, training loop, FP8 filter, logging |
| `nanochat/loss_eval.py` | Handle `(loss, aux_loss)` tuple return from MoE model |
| `runs/tmp_moe.sh` | New launch script for MoE training |

### GPTConfig New Fields

```python
use_moe: bool = False
n_experts: int = 8
expert_topk: int = 2
moe_freq: int = 1       # every N-th layer uses MoE (1=all layers)
expert_hidden_mult: int = 4  # expert hidden dim = expert_hidden_mult * n_embd
```

### Iso-FLOPs Design

To compare MoE vs dense on equal compute budgets:
- Dense baseline: hidden dim = `4 * n_embd` (hardcoded in `gpt.py:MLP`)
- MoE setting: `expert_hidden_mult=2`, `expert_topk=2` => active hidden dim = `2 * 2 * n_embd = 4 * n_embd`
- This ensures the per-token FLOPs of MoE match the dense baseline exactly

### Key Design Decisions

1. **Conditional return type**: `GPT.forward()` returns `(loss, aux_loss)` when `use_moe=True` + targets given; returns plain `loss` otherwise. This preserves backward compatibility.

2. **Internal tuple interface**: `Block.forward()` always returns `(x, aux_loss)`. Dense blocks return `aux_loss=0.0`.

3. **Gate initialization**: Zeros. `softmax(zeros) = uniform` distribution, so all experts receive equal routing at init.

4. **Expert initialization**: Same as dense MLP (`c_fc`: uniform, `c_proj`: zeros).

5. **Optimizer**: MoE params (gate + experts) live inside `self.transformer.h`, so they automatically join the Muon param group. No optimizer code changes needed.

6. **Balance loss**: Standard Switch Transformer formula: `L_balance = N * sum(f_i * P_i)` where `f_i` is expert frequency and `P_i` is mean routing probability. Scaled by `--balance-loss-coeff` (default 0.01).

7. **Active parameter counting**: `num_scaling_params()` reports `moe_inactive` (params in non-selected experts) and `active_total`. `estimate_flops()` excludes inactive expert params. `get_scaling_params()` in the training script subtracts `moe_inactive` for compute-optimal token budget calculation.

---

## Current Status

### Training Configuration (test run)
```
Model: 12 layers, aspect_ratio=64, head_dim=64
MoE: 8 experts, top-2, expert_hidden_mult=2
Sequence length: 1024
Device batch size: 16 (8x H800 GPUs)
FP8: enabled (excluding MoE expert layers)
Balance loss coeff: 0.01
```

### Parameter Counts
- **Total params**: ~456M
- **Active params (per token)**: ~286M
- **MoE inactive**: ~170M (experts not selected by router)

### Training Results (completed, 1733 steps)
- Cross-entropy loss: 10.18 -> 2.99 (smoothed 3.01)
- Balance loss: stable ~24.0 throughout training
- MFU: ~8.3% (steady state)
- Total training time: 25.25 min (8x H800)
- Total training FLOPs: 7.04e17
- Checkpoint saved: `test_moe_small/model_001733.pt`

### Evaluation Results (BPB & CORE)
- **Train BPB**: 0.9012
- **Val BPB**: 0.9015
- **CORE metric**: 0.1116

Selected benchmark scores:
| Benchmark | Accuracy |
|-----------|----------|
| HellaSwag (0-shot) | 0.327 |
| ARC-Easy (10-shot) | 0.514 |
| ARC-Challenge (10-shot) | 0.257 |
| PIQA (10-shot) | 0.645 |
| COPA (0-shot) | 0.570 |
| Winograd (0-shot) | 0.586 |
| BoolQ (10-shot) | 0.470 |
| LAMBADA (0-shot) | 0.273 |

---

## 2026-02-24 Routing Fastpath Study

### What changed
- Reworked `nanochat/moe.py` dispatch path to group token assignments by expert and accumulate outputs via `index_add_`, replacing repeated boolean-mask scans.
- Added input validation (`1 <= expert_topk <= n_experts`) in `MoE.__init__`.
- Removed inline debug-only routines from `nanochat/moe.py` and moved verification into proper unit tests.
- Added `tests/test_moe.py` with:
  - numerical equivalence against reference boolean-mask routing
  - gradient-flow checks for gate and experts
  - invalid-configuration guard test
- Added reusable benchmark script `dev/bench_moe_routing.py`.

### Validation
- Unit tests:
  - `.venv/bin/python -m pytest tests/test_moe.py -v` -> 3 passed
- Microbenchmark (single GPU, H800, BF16):
  - Shape: `B=16, T=1024, D=768, experts=8, topk=2`
  - Reference path: `2.538 ms`
  - New path: `1.106 ms`
  - Speedup: `2.295x`
  - Output diff: `0.0`, aux diff: `0.0`
- 8-GPU smoke (`torchrun --nproc_per_node=8`, random-input train steps):
  - world size: 8
  - global mean step latency: `13.299 ms`
  - final loss: `0.020386`
  - final aux: `2.000000`
  - status: completed without NaN/divergence

### Artifacts
- `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_072951_moe-routing-microbench/`
- `/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp/20260224_073107_moe-routing-ddp-smoke/`

---

## Dense vs MoE Iso-FLOPs Comparison

### Training Configuration (identical except MoE flags)
Both models: 12 layers, aspect_ratio=64, head_dim=64, seq_len=1024, 8x H800, FP8, target_param_data_ratio=8.25

| | Dense (gpt2_small) | MoE (moe_small) |
|---|---|---|
| Total params | 286M | 456M |
| Active params/token | 286M | 286M |
| Training tokens | 908M | 908M |
| Training time | **5.34 min** | 25.25 min |
| MFU | **~36%** | ~8% |

### Quality Comparison
| Metric | Dense | MoE | Delta |
|--------|-------|-----|-------|
| **Val BPB** | 0.9161 | **0.9015** | **-1.6%** |
| **Train BPB** | 0.9160 | **0.9012** | -1.6% |
| **CORE metric** | 0.1113 | **0.1116** | +0.03% |
| **Final loss** | 3.06 | **2.99** | -2.3% |

### Benchmark Details (final eval)
| Benchmark | Dense | MoE | Delta |
|-----------|-------|-----|-------|
| HellaSwag (0-shot) | 0.315 | **0.327** | +0.012 |
| ARC-Easy (10-shot) | 0.493 | **0.514** | +0.021 |
| ARC-Challenge (10-shot) | 0.247 | **0.257** | +0.010 |
| PIQA (10-shot) | 0.612 | **0.645** | +0.033 |
| COPA (0-shot) | **0.580** | 0.570 | -0.010 |
| Winograd (0-shot) | 0.586 | 0.586 | 0.000 |
| Winogrande (0-shot) | **0.509** | 0.503 | -0.006 |
| BoolQ (10-shot) | **0.502** | 0.470 | -0.032 |
| LAMBADA (0-shot) | 0.247 | **0.273** | +0.026 |
| CommonsenseQA (10-shot) | 0.238 | **0.270** | +0.032 |

### Conclusion
MoE wins on perplexity (BPB -1.6%) and most benchmarks (PIQA +3.3%, LAMBADA +2.6%, ARC-Easy +2.1%). CORE composite is essentially tied. The cost is ~5x slower wall-clock training due to unoptimized MoE kernel (sequential expert loop, no FP8 on experts, torch.compile recompilation).

---

## Experimental Observations

### Balance Loss
- Stable around 24.0 throughout training (start to finish)
- Formula: `N_experts * sum(f_i * P_i)` where perfect balance gives `N_experts * (topk/N_experts)^2 * N_experts = topk^2 / N_experts * N_experts^2`
- For 8 experts, top-2: theoretical minimum = `8 * 8 * (1/8)^2 = 8 * (2/8) = 2.0` (if perfectly uniform)
- Observed 24.0 suggests some expert specialization is emerging, which is expected

### MFU (~8%)
MFU is lower than typical dense models (~30%+) due to:
1. **Dynamic shapes**: MoE routing with boolean indexing (`x_flat[mask.any(dim=-1)]`) produces variable-size tensors each step
2. **torch.compile recompilation**: Hit `recompile_limit` (8) for MoE forward, then falls back to eager mode
3. **Expert loop**: Current implementation loops over experts sequentially (not batched/fused)
4. MFU calculation uses total params, but only `topk/n_experts` fraction is active

### FP8 Compatibility
FP8 conversion works for attention layers but fails for MoE expert layers due to dynamic shapes. The `max(abs(input))` operation in FP8 scaling fails on empty tensors (when no tokens are routed to an expert).

---

## Known Issues

### 1. FP8 + MoE Expert Layers
**Problem**: FP8 scaling computes `max(abs(input))` which fails on zero-dimension tensors. MoE routing can produce `[0, hidden_dim]` shaped inputs when no tokens route to an expert.

**Workaround**: Skip MoE expert layers in `fp8_module_filter`:
```python
if 'experts' in fqn:
    return False
```
Result: 49 layers converted to FP8 (attention), 210 skipped (experts + small dims).

**Proper fix**: Either pad expert inputs to avoid empty tensors, or use a fused MoE kernel (like Megablocks) that handles sparsity natively.

### 2. torch.compile Recompilation
**Problem**: `torch._dynamo` hits `recompile_limit` (8) for MoE forward due to changing tensor shapes from top-k routing.

**Impact**: Performance degradation (contributes to low MFU). The MoE forward eventually falls back to eager execution.

**Proper fix**: Use `torch.compile(dynamic=True)` for MoE layers, or switch to a token-permutation approach that produces fixed-size tensors.

### 3. Sequential Expert Loop
**Problem**: Current MoE implementation loops over experts one by one in Python:
```python
for idx in range(self.n_experts):
    mask = (expert_indices == idx)
    x_idx = x_flat[mask.any(dim=-1)]
    out[mask.any(dim=-1)] += self.experts[idx](x_idx) * routing_weights[mask].unsqueeze(-1)
```

**Impact**: Cannot benefit from GPU parallelism across experts.

**Proper fix**: Use a grouped/batched GEMM kernel (e.g., Megablocks, vLLM's fused MoE, or Triton-based kernels).

---

## Next Steps

### 1. Shared Experts (DeepSeek-style)
Add `n_shared_experts` parameter. Shared experts process all tokens (no routing) and their output is added to the routed experts' output. This provides a "common knowledge" pathway that stabilizes training.

Key design points:
- Shared experts use the same `MLP` class but are always active
- Output: `shared_out + routed_out`
- FLOPs accounting must include shared expert compute

### 2. Sigmoid Routing (DeepSeek-V3 style)
Replace softmax gating with sigmoid + top-k selection:
- `gate_scores = sigmoid(gate_logits)` instead of `softmax(gate_logits)`
- Select top-k experts per token
- Normalize selected scores: `weights = scores / scores.sum()`

Benefits:
- Decouples expert selection from scoring (softmax creates competition between experts)
- More stable training, especially with many experts
- Naturally supports auxiliary-loss-free training with bias terms

### 3. Performance Optimization
- Investigate grouped GEMM / fused MoE kernels for better GPU utilization
- Consider capacity factor + token dropping for fixed-shape tensors (enables FP8 + torch.compile)
- Profile to identify the actual bottleneck between compute and memory bandwidth
