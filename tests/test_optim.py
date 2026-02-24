"""
Tests for optimizer edge cases.
"""

import torch

import nanochat.optim as optim_mod
from nanochat.optim import MuonAdamW, _grad_or_zeros


def test_grad_or_zeros_handles_none_grad():
    p = torch.nn.Parameter(torch.randn(3, 3))
    p.grad = None

    grad = _grad_or_zeros(p)

    assert grad.shape == p.shape
    assert grad.dtype == p.dtype
    assert grad.device == p.device
    assert torch.count_nonzero(grad) == 0


def test_muon_step_handles_none_grad(monkeypatch):
    p0 = torch.nn.Parameter(torch.randn(4, 4))
    p1 = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = MuonAdamW([
        dict(
            kind="muon",
            params=[p0, p1],
            lr=0.1,
            momentum=0.95,
            ns_steps=1,
            beta2=0.95,
            weight_decay=0.0,
        )
    ])
    p0.grad = torch.randn_like(p0)
    p1.grad = None

    captured = {}

    def fake_muon_step_fused(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_momentum_buffer,
        momentum_t,
        lr_t,
        wd_t,
        beta2_t,
        ns_steps,
        red_dim,
    ):
        captured["stacked_grads"] = stacked_grads.detach().clone()
        stacked_params.add_(stacked_grads, alpha=-0.01)

    monkeypatch.setattr(optim_mod, "muon_step_fused", fake_muon_step_fused)
    p1_before = p1.detach().clone()

    optimizer.step()

    assert "stacked_grads" in captured
    assert torch.allclose(captured["stacked_grads"][1], torch.zeros_like(p1))
    assert torch.allclose(p1, p1_before)
