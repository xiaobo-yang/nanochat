import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.expert_hidden_mult * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.expert_hidden_mult * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_experts)])
        self.topk = config.expert_topk
        self.n_experts = config.n_experts
        self.capacity_factor = float(getattr(config, "moe_capacity_factor", 0.0))
        if not (1 <= self.topk <= self.n_experts):
            raise ValueError(
                f"expert_topk must satisfy 1 <= expert_topk <= n_experts, "
                f"got expert_topk={self.topk}, n_experts={self.n_experts}"
            )
        if self.capacity_factor < 0:
            raise ValueError(
                f"moe_capacity_factor must be >= 0, got {self.capacity_factor}"
            )

    def _dispatch_exact(
        self,
        x_flat: torch.Tensor,
        token_indices: torch.Tensor,
        expert_indices_flat: torch.Tensor,
        routing_weights_flat: torch.Tensor,
    ):
        sort_perm = torch.argsort(expert_indices_flat)
        token_indices_sorted = token_indices[sort_perm]
        routing_weights_sorted = routing_weights_flat[sort_perm]
        expert_counts = torch.bincount(expert_indices_flat, minlength=self.n_experts)

        out = torch.zeros_like(x_flat)
        start = 0
        for idx, count in enumerate(expert_counts.tolist()):
            if count == 0:
                continue
            end = start + count
            idx_tokens = token_indices_sorted[start:end]
            idx_weights = routing_weights_sorted[start:end]
            expert_out = self.experts[idx](x_flat.index_select(0, idx_tokens))
            out.index_add_(0, idx_tokens, expert_out * idx_weights.unsqueeze(-1).to(expert_out.dtype))
            start = end
        return out, expert_counts

    def _dispatch_fixed_capacity(
        self,
        x_flat: torch.Tensor,
        token_indices: torch.Tensor,
        expert_indices_flat: torch.Tensor,
        routing_weights_flat: torch.Tensor,
    ):
        num_assignments = token_indices.numel()
        capacity = max(
            1,
            math.ceil((num_assignments / self.n_experts) * self.capacity_factor),
        )

        sort_perm = torch.argsort(expert_indices_flat)
        token_indices_sorted = token_indices[sort_perm]
        routing_weights_sorted = routing_weights_flat[sort_perm]
        expert_counts = torch.bincount(expert_indices_flat, minlength=self.n_experts)

        out = torch.zeros_like(x_flat)
        start = 0
        # Allocate per-forward padded routing buffers once, then reuse per expert.
        all_tokens = torch.zeros(
            self.n_experts, capacity, dtype=token_indices.dtype, device=token_indices.device
        )
        all_weights = torch.zeros(
            self.n_experts, capacity, dtype=routing_weights_sorted.dtype, device=routing_weights_sorted.device
        )
        dropped = 0
        for idx, count in enumerate(expert_counts.tolist()):
            end = start + count
            take = min(count, capacity)
            if take > 0:
                all_tokens[idx, :take] = token_indices_sorted[start:start + take]
                all_weights[idx, :take] = routing_weights_sorted[start:start + take]

            dropped += count - take
            start = end

        for idx in range(self.n_experts):
            idx_tokens = all_tokens[idx]
            idx_weights = all_weights[idx]
            expert_out = self.experts[idx](x_flat.index_select(0, idx_tokens))
            out.index_add_(0, idx_tokens, expert_out * idx_weights.unsqueeze(-1).to(expert_out.dtype))

        self.last_dropped_assignments = dropped
        self.last_drop_fraction = dropped / max(1, num_assignments)
        self.last_capacity = capacity
        return out, expert_counts

    def forward(self, x):  # x is (B, T, D)
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B*T, E)
        routing_weights, expert_indices = gate_probs.topk(k=self.topk, dim=-1)  # (B*T, topk)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        num_tokens = B * T
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(-1).expand(-1, self.topk).reshape(-1)
        expert_indices_flat = expert_indices.reshape(-1)
        routing_weights_flat = routing_weights.reshape(-1)

        if self.capacity_factor > 0:
            out, expert_counts = self._dispatch_fixed_capacity(
                x_flat, token_indices, expert_indices_flat, routing_weights_flat
            )
        else:
            out, expert_counts = self._dispatch_exact(
                x_flat, token_indices, expert_indices_flat, routing_weights_flat
            )

        expert_freq = expert_counts.to(dtype=gate_probs.dtype) / num_tokens
        out = out.view(B, T, D)
        # loss = E * sum_i f_i * P_i, where P_i is mean routing probability of expert i
        balance_loss = (gate_probs.mean(dim=0) * expert_freq.detach()).sum() * self.n_experts
        return out, balance_loss
