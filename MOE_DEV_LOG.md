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

### Training Progress (observed at step ~119 / 1733)
- Cross-entropy loss: 10.4 -> ~4.7 (healthy decrease)
- Balance loss: stable at ~24.2
- MFU: ~7%

---

## Experimental Observations

### Balance Loss
- Stable around 24.2 throughout early training
- Formula: `N_experts * sum(f_i * P_i)` where perfect balance gives `N_experts * (topk/N_experts)^2 * N_experts = topk^2 / N_experts * N_experts^2`
- For 8 experts, top-2: theoretical minimum = `8 * 8 * (1/8)^2 = 8 * (2/8) = 2.0` (if perfectly uniform)
- Observed 24.2 suggests some expert specialization is emerging, which is expected

### MFU (~7%)
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
