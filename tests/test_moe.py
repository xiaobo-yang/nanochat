"""
Tests for MoE routing, stability, and gradients.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nanochat.moe import MoE


@dataclass
class DummyConfig:
    n_embd: int = 16
    n_experts: int = 4
    expert_topk: int = 2
    expert_hidden_mult: int = 2


def _reference_forward(model: MoE, x: torch.Tensor):
    """Reference implementation matching the original boolean-mask dispatch."""
    B, T, D = x.shape
    x_flat = x.reshape(B * T, D)
    gate_logits = model.gate(x_flat)
    gate_probs = F.softmax(gate_logits, dim=-1)
    routing_weights, expert_indices = gate_probs.topk(k=model.topk, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(x_flat)
    expert_freq = torch.zeros(model.n_experts, dtype=gate_probs.dtype, device=x.device)
    for idx in range(model.n_experts):
        selected = (expert_indices == idx).any(dim=-1)
        if selected.any():
            weights = routing_weights[expert_indices == idx].unsqueeze(-1)
            out[selected] += model.experts[idx](x_flat[selected]) * weights.to(out.dtype)
        expert_freq[idx] = selected.float().mean()

    balance_loss = (gate_probs.mean(dim=0) * expert_freq).sum() * model.n_experts
    return out.view(B, T, D), balance_loss


def test_moe_matches_reference_forward():
    torch.manual_seed(1234)
    config = DummyConfig()
    model = MoE(config)
    x = torch.randn(3, 5, config.n_embd)

    out_ref, aux_ref = _reference_forward(model, x)
    out_new, aux_new = model(x)

    assert torch.allclose(out_new, out_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(aux_new, aux_ref, atol=1e-6, rtol=1e-5)


def test_moe_backward_has_gate_and_expert_gradients():
    torch.manual_seed(42)
    config = DummyConfig()
    model = MoE(config)
    x = torch.randn(2, 7, config.n_embd)

    out, aux_loss = model(x)
    loss = out.pow(2).mean() + 0.1 * aux_loss
    loss.backward()

    gate_grad = model.gate.weight.grad
    assert gate_grad is not None
    assert gate_grad.abs().sum().item() > 0

    grad_sums = []
    for expert in model.experts:
        grad = expert.c_fc.weight.grad
        if grad is None:
            continue
        grad_sums.append(grad.abs().sum().item())
    assert any(grad_sum > 0 for grad_sum in grad_sums)


def test_moe_rejects_invalid_topk():
    bad = DummyConfig(n_experts=4, expert_topk=5)
    try:
        MoE(bad)
    except ValueError as exc:
        assert "expert_topk" in str(exc)
    else:
        raise AssertionError("MoE must reject expert_topk > n_experts")
