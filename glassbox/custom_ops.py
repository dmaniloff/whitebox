"""
Custom torch operations for graph instrumentation.

This module defines custom operations that can be used to instrument
torch graphs during compilation passes.
"""

from typing import Tuple

import torch

from .svd import matvec_S, matvec_ST, randomized_svd, svd_via_lanczos


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
    _q_grouped = q.view(
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


@torch.library.custom_op("glassbox::svd_of_scores_matrix_rnd", mutates_args=())
def svd_of_scores_matrix_rnd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom op to compute (and print) singular values of the scores matrix S = QK^T.

    This is a passthrough op intended for instrumentation. It runs a small
    randomized SVD on a single representative head (to keep overhead modest),
    prints the resulting singular values, and returns cloned inputs.
    """
    with torch.no_grad():
        n = q.shape[0]  # flattened sequence length
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        d = q.shape[2]

        if num_kv_heads > 0 and num_q_heads % num_kv_heads == 0:
            q_grouped = q.view(n, num_kv_heads, num_q_heads // num_kv_heads, d)
            Qh = q_grouped[:, 0, 0, :]
            Kh = k[:, 0, :]
        else:
            # Fallback: just take the first heads.
            Qh = q[:, 0, :]
            Kh = k[:, 0, :]

        # Run the SVD math in fp32 to avoid dtype mismatches (e.g. bf16 Q/K with fp32 Omega)
        # and to keep torch.linalg.qr/svd on a well-supported dtype.
        Qh = Qh.float()
        Kh = Kh.float()

        def _matvec(x: torch.Tensor) -> torch.Tensor:
            return matvec_S(Qh, Kh, x)

        def _matvec_t(x: torch.Tensor) -> torch.Tensor:
            return matvec_ST(Qh, Kh, x)

        k_rank = int(min(8, n))
        U, S, V = randomized_svd(
            matvec=_matvec,
            matvec_t=_matvec_t,
            dim=int(n),
            k=k_rank,
            p=4,
            q=1,
            device=str(q.device),
        )
        _ = (U, V)  # silence unused warnings in some tooling
        s_list = [float(x) for x in S.detach().to("cpu").tolist()]
        print(f"[SVD_RND] {layer_name} S(QK^T) top-{k_rank}: {s_list}")

    return q.clone(), k.clone(), v.clone()


@svd_of_scores_matrix_rnd.register_fake
def _(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return q.clone(), k.clone(), v.clone()


@torch.library.custom_op("glassbox::svd_of_scores_matrix_lanczos", mutates_args=())
def svd_of_scores_matrix_lanczos(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom op to compute (and print) singular values of the scores matrix S = QK^T.

    This is a passthrough op intended for instrumentation. It runs a small
    Lanczos-based SVD (via Lanczos on M^T M) on a single representative head,
    prints the resulting singular values, and returns cloned inputs.
    """
    with torch.no_grad():
        n = q.shape[0]
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        d = q.shape[2]

        if num_kv_heads > 0 and num_q_heads % num_kv_heads == 0:
            q_grouped = q.view(n, num_kv_heads, num_q_heads // num_kv_heads, d)
            Qh = q_grouped[:, 0, 0, :]
            Kh = k[:, 0, :]
        else:
            Qh = q[:, 0, :]
            Kh = k[:, 0, :]

        Qh = Qh.float()
        Kh = Kh.float()

        def _matvec(x: torch.Tensor) -> torch.Tensor:
            return matvec_S(Qh, Kh, x)

        def _matvec_t(x: torch.Tensor) -> torch.Tensor:
            return matvec_ST(Qh, Kh, x)

        k_rank = int(min(8, n))
        U, S, V = svd_via_lanczos(
            matvec=_matvec,
            matvec_t=_matvec_t,
            dim=int(n),
            k=k_rank,
            iters=20,
            device=str(q.device),
        )
        _ = (U, V)
        s_list = [float(x) for x in S.detach().to("cpu").tolist()]
        print(f"[SVD_LANCZOS] {layer_name} S(QK^T) top-{k_rank}: {s_list}")

    return q.clone(), k.clone(), v.clone()


@svd_of_scores_matrix_lanczos.register_fake
def _(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return q.clone(), k.clone(), v.clone()
