"""
Benchmark MoE routing dispatch implementation against a reference boolean-mask path.

Usage:
python dev/bench_moe_routing.py --device cuda --iters 60 --warmup 15
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

# Support direct execution: `python dev/bench_moe_routing.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanochat.moe import MoE


@dataclass
class Config:
    n_embd: int = 768
    n_experts: int = 8
    expert_topk: int = 2
    expert_hidden_mult: int = 2


def reference_forward(model: MoE, x: torch.Tensor):
    """Original per-expert boolean-mask routing implementation."""
    B, T, D = x.shape
    x_flat = x.reshape(B * T, D)
    gate_logits = model.gate(x_flat)
    gate_probs = F.softmax(gate_logits, dim=-1)
    routing_weights, expert_indices = gate_probs.topk(k=model.topk, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(x_flat)
    expert_freq = torch.zeros(model.n_experts, dtype=gate_probs.dtype, device=x.device)
    for idx in range(model.n_experts):
        mask = expert_indices == idx
        token_mask = mask.any(dim=-1)
        if token_mask.any():
            expert_in = x_flat[token_mask]
            expert_out = model.experts[idx](expert_in)
            out[token_mask] += expert_out * routing_weights[mask].unsqueeze(-1).to(expert_out.dtype)
        expert_freq[idx] = token_mask.float().mean()

    balance_loss = (gate_probs.mean(dim=0) * expert_freq).sum() * model.n_experts
    return out.view(B, T, D), balance_loss


def benchmark(fn, model: MoE, x: torch.Tensor, warmup: int, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn(model, x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(model, x)
        if x.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MoE routing forward path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="benchmark device")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"], help="tensor dtype")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU benchmark currently supports float32 only")

    cfg = Config()
    model = MoE(cfg).to(device=device, dtype=dtype)
    model.eval()
    x = torch.randn(args.batch_size, args.seq_len, cfg.n_embd, device=device, dtype=dtype)

    with torch.inference_mode():
        out_ref, aux_ref = reference_forward(model, x)
        out_new, aux_new = model(x)
    max_abs_diff = (out_ref - out_new).abs().max().item()
    aux_abs_diff = abs(aux_ref.item() - aux_new.item())

    ref_ms = benchmark(reference_forward, model, x, args.warmup, args.iters)
    new_ms = benchmark(lambda m, y: m(y), model, x, args.warmup, args.iters)

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"shape=B{args.batch_size}xT{args.seq_len}xD{cfg.n_embd}, experts={cfg.n_experts}, topk={cfg.expert_topk}")
    print(f"reference_ms={ref_ms:.3f}")
    print(f"new_ms={new_ms:.3f}")
    print(f"speedup_x={ref_ms / new_ms:.3f}")
    print(f"max_abs_diff={max_abs_diff:.6e}")
    print(f"aux_abs_diff={aux_abs_diff:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
