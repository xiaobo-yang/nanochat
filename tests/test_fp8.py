"""
FP8 edge-case tests.
"""

import pytest
import torch

from nanochat.fp8 import Float8Linear, _to_fp8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for FP8 dtype path")
def test_to_fp8_handles_empty_tensor():
    x = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
    x_fp8, inv = _to_fp8(x, torch.float8_e4m3fn)
    assert x_fp8.shape == x.shape
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert inv.shape == torch.Size([])
    assert inv.item() == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for FP8 matmul")
def test_float8_linear_forward_allows_empty_batch():
    linear = torch.nn.Linear(64, 32, bias=False, device="cuda", dtype=torch.bfloat16)
    fp8_linear = Float8Linear.from_float(linear).to(device="cuda")
    x = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
    y = fp8_linear(x)
    assert y.shape == (0, 32)
