import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

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

    def forward(self, x): # x is (B, T, D)
        B, T, D = x.shape
        x_flat = x.view(B*T, -1)
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1) # (B*T, E)
        routing_weights, expert_indices = gate_probs.topk(k=self.topk, dim=-1) # (B*T, topk)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True) 
        
        out = torch.zeros_like(x_flat, dtype=x.dtype, device=x.device)
        expert_freq = torch.zeros(self.n_experts, dtype=x.dtype, device=x.device)
        for idx in range(self.n_experts):
            mask = (expert_indices == idx)
            x_idx = x_flat[mask.any(dim=-1)]
            out[mask.any(dim=-1)] += self.experts[idx](x_idx) * routing_weights[mask].unsqueeze(-1)
            expert_freq[idx] = mask.any(dim=-1).float().mean().detach()
        out = out.view(B, T, -1)
        # loss = E \sum_i f_i x P_i, P_i is mean of routing probability of expert i
        balance_loss = (gate_probs.mean(dim=0) * expert_freq).sum() * self.n_experts 
        
        return out, balance_loss


# ------------------------------------------------------------
# debug
def test_moe_gradients():
    # 1. 配置极简环境
    config = type('Config', (), {
        'n_embd': 16,
        'n_experts': 4,
        'expert_topk': 2
    })()
    
    model = MoE(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 2. 构造 Dummy Data
    x = torch.randn(2, 8, config.n_embd) # (B, T, D)
    
    # 3. 前向传播
    out, aux_loss = model(x)
    loss = out.mean() + aux_loss
    
    # 4. 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 5. 【关键】检查梯度是否存在
    print("--- Gradient Check ---")
    
    # 检查 Gate 是否有梯度 (最容易出错的地方)
    if model.gate.weight.grad is None or model.gate.weight.grad.abs().sum() == 0:
        print("❌ CRITICAL: Gate has NO gradient! Routing is broken.")
    else:
        print(f"✅ Gate gradient magnitude: {model.gate.weight.grad.abs().mean().item():.6f}")

    # 检查专家是否有梯度
    expert_grad = model.experts[0].c_fc.weight.grad
    if expert_grad is None or expert_grad.abs().sum() == 0:
        print("⚠️ Warning: Expert 0 has no gradient (Might be due to randomness, check others)")
    else:
        print(f"✅ Expert 0 gradient detected.")

def test_moe_overfit():
    torch.manual_seed(42) # 保证可复现
    
    config = type('Config', (), {
        'n_embd': 32,
        'n_experts': 4,
        'expert_topk': 2
    })()
    
    model = MoE(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 构造固定的“伪数据”
    # 任务：让模型记住输入 x 对应的目标 y
    x = torch.randn(4, 16, config.n_embd) 
    y = torch.randn(4, 16, config.n_embd) 
    
    print("\n--- Overfit Test ---")
    for step in range(201):
        out, aux_loss = model(x)
        fit_loss = F.mse_loss(out, y) # 简单的回归任务
        loss = fit_loss + 0.1 * aux_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}, Fit Loss: {fit_loss.item():.6f}, Aux Loss: {aux_loss.item():.6f}")

            with torch.no_grad():
                # 再次跑一遍前向，或者让 forward 返回 gate_probs
                # 这里假设我们手动提取一下逻辑用来测试
                x_flat = x.view(-1, config.n_embd)
                logits = model.gate(x_flat)
                probs = F.softmax(logits, dim=-1)
                
                # 打印每个专家平均分配到的概率
                expert_usage = probs.mean(dim=0) 
                print(f"Expert Usage: {expert_usage.tolist()}") 
                _, expert_indices = probs.topk(k=config.expert_topk, dim=-1)
                expert_freq = torch.zeros(config.n_experts, dtype=x.dtype, device=x.device)
                for idx in range(config.n_experts):
                    mask = (expert_indices == idx)
                    expert_freq[idx] = mask.any(dim=-1).float().mean().detach()
                print(f"Expert Freq: {expert_freq.tolist()}")
                # 理想情况：[0.25, 0.25, 0.25, 0.25] 左右
                # 糟糕情况：[0.99, 0.00, 0.01, 0.00] -> 需要 Load Balancing Loss!
            
    if fit_loss.item() < 0.01:
        print("✅ PASSED: Model can overfit random data.")
    else:
        print("❌ FAILED: Model cannot learn. Check architecture.")

if __name__ == "__main__":
    test_moe_overfit()
    test_moe_gradients()

        

         

