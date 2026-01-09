"""
Custom torch operations for graph instrumentation.

This module defines custom operations that can be used to instrument
torch graphs during compilation passes.
"""

from typing import Tuple

import torch


# Register a custom torch operation for capturing QKV values before attention
@torch.library.custom_op("glassbox::capture_qkv", mutates_args=())
def capture_qkv_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom op to capture Q, K, V values before attention.

    This is a passthrough operation that logs statistics about the QKV tensors
    while preserving the original tensor data.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        layer_name: Name identifier for the layer being captured

    Returns:
        A tuple of clones of the input tensors (q, k, v)
    """
    print(
        f"[QKV_CAPTURE] {layer_name} - Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}"
    )
    n = q.shape[0]  # flattened sequence length
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    d = q.shape[2]

    # Q: [8192, 32, 128], K: [8192, 8, 128]
    # Reshape Q: 32 heads → 8 groups × 4 heads per group
    q_grouped = q.view(
        n, num_kv_heads, num_q_heads // num_kv_heads, d
    )  # [n, num_kv_heads, heads_per_group, d]

    # logits = torch.einsum("ngqd,mgd->gnqm", q_grouped, k)
    # Result: [8, 8192, 4, 8192] = [kv_heads, seq, q_per_group, seq]

    # reshape to standard form:
    # logits = logits.permute(0, 2, 1, 3).reshape(num_q_heads, n, n)
    # Result: [32, 8192, 8192]

    # print(f"[QKV_CAPTURE] {layer_name} - Logits shape: {q_grouped.shape}")

    return q.clone(), k.clone(), v.clone()


@capture_qkv_op.register_fake
def _(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake implementation for torch.compile tracing.

    Output has same shapes/dtypes/devices as inputs.
    """
    return q.clone(), k.clone(), v.clone()


# Register a custom torch operation for capturing mean values
@torch.library.custom_op("glassbox::capture_mean", mutates_args=())
def capture_mean_op(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    """
    Custom op to capture mean values.

    This is a passthrough operation that logs the mean of the tensor
    while preserving the original tensor data.

    Args:
        x: Input tensor
        layer_name: Name identifier for the layer being captured

    Returns:
        A clone of the input tensor
    """
    mean_val = float(x.mean().item())
    print(f"[MEAN_CAPTURE] {layer_name} attention mean: {mean_val:.6f}")
    return x.clone()  # Passthrough


# Register the fake/abstract implementation for torch.compile
@capture_mean_op.register_fake
def _(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    """
    Fake implementation for torch.compile tracing.

    Output has same shape/dtype/device as input.
    """
    return x.clone()
