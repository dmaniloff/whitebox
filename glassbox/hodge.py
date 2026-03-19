"""
Hodge decomposition safety features for the degree-normalized cross-operator M.

Computes asymmetry coefficient G, curl estimate C, Pythagorean decomposition
Gamma, and curl_ratio from either a materialized M or matrix-free via Q, K.
"""

import math

import numpy as np
import torch

EPSILON = 1e-10

from glassbox.svd import (
    compute_logsumexp_blocked,
    compute_M_fro_norm_blocked,
    get_M_entries_batch,
    matvec_M_blocked,
    matvec_MT_blocked,
    randomized_svd,
    svd_via_lanczos,
)


def sample_triangles(n, n_samples, seed=42):
    """Random triangle sampling. Returns (n_samples, 3) int64 with i < j < k."""
    if n < 3 or n_samples <= 0:
        return np.zeros((0, 3), dtype=np.int64)
    rng = np.random.RandomState(seed)
    total = n * (n - 1) * (n - 2) // 6
    n_samples = min(n_samples, total)
    triangles = set()
    while len(triangles) < n_samples:
        batch = rng.randint(0, n, size=(n_samples * 3, 3))
        for row in batch:
            tri = tuple(sorted(row))
            if tri[0] < tri[1] < tri[2]:
                triangles.add(tri)
                if len(triangles) >= n_samples:
                    break
    result = np.array(list(triangles)[:n_samples], dtype=np.int64)
    return result


def adaptive_curl_samples(n, target_cv=0.05):
    """Formula-based adaptive sample count for curl estimation."""
    if n < 4:
        return 0
    total = n * (n - 1) * (n - 2) // 6
    # Cochran-like formula: n_samples = 1 / (target_cv^2)
    n_samples = int(math.ceil(1.0 / (target_cv**2)))
    return min(n_samples, total)


def estimate_curl_materialized(M, target_cv=0.05, seed=42):
    """Triangle-sampling curl on a materialized M tensor."""
    n = M.shape[0]
    n_samp = adaptive_curl_samples(n, target_cv)
    if n_samp == 0:
        return 0.0
    tri = sample_triangles(n, n_samp, seed)
    ii = torch.from_numpy(tri[:, 0]).to(M.device)
    jj = torch.from_numpy(tri[:, 1]).to(M.device)
    kk = torch.from_numpy(tri[:, 2]).to(M.device)
    circs = (M[ii, jj] - M[jj, ii]) + (M[jj, kk] - M[kk, jj]) - (M[ii, kk] - M[kk, ii])
    rms = circs.square().mean().sqrt()
    M_fro_norm = torch.linalg.norm(M, "fro").item()
    C = rms.item() / (math.sqrt(2) * (M_fro_norm + EPSILON))
    return C


def estimate_curl_matrix_free(
    Q, K, lse, d_k_inv_sqrt, scale, M_fro_norm, target_cv=0.05, seed=42
):
    """Triangle-sampling curl using on-the-fly M[i,j] lookups."""
    n = Q.shape[0]
    n_samp = adaptive_curl_samples(n, target_cv)
    if n_samp == 0:
        return 0.0
    tri = sample_triangles(n, n_samp, seed)
    ii = torch.from_numpy(tri[:, 0]).to(Q.device)
    jj = torch.from_numpy(tri[:, 1]).to(Q.device)
    kk = torch.from_numpy(tri[:, 2]).to(Q.device)

    def M_entry(a, b):
        return get_M_entries_batch(Q, K, lse, d_k_inv_sqrt, scale, a, b)

    circs = (
        (M_entry(ii, jj) - M_entry(jj, ii))
        + (M_entry(jj, kk) - M_entry(kk, jj))
        - (M_entry(ii, kk) - M_entry(kk, ii))
    )
    rms = circs.square().mean().sqrt()
    C = rms.item() / (math.sqrt(2) * (M_fro_norm + EPSILON))
    return C


def compute_G_materialized(M):
    """Asymmetry coefficient: G = ||M_asym||_F / (||M||_F + eps)."""
    M_fro = torch.linalg.norm(M, "fro")
    M_asym = (M - M.T) / 2.0
    M_asym_fro = torch.linalg.norm(M_asym, "fro")
    G = (M_asym_fro / (M_fro + EPSILON)).item()
    return G, M_fro.item()


def compute_G_matrix_free(Q, K, d_k_inv_sqrt, scale, block_size=256):
    """Matrix-free G via blocked streaming.

    Computes ||M||_F^2 and <M, M^T>_F in one pass over row blocks.
    ||M_asym||_F^2 = (||M||_F^2 - <M, M^T>_F) / 2
    """
    L = Q.shape[0]
    lse = compute_logsumexp_blocked(Q, K, scale, block_size)
    norm_sq = torch.tensor(0.0, device=Q.device, dtype=Q.dtype)
    inner_sq = torch.tensor(0.0, device=Q.device, dtype=Q.dtype)

    for i0 in range(0, L, block_size):
        i1 = min(i0 + block_size, L)
        bs = i1 - i0
        scores = Q[i0:i1] @ K.T * scale
        attn = torch.softmax(scores, dim=-1)  # [bs, L]
        M_block = attn * d_k_inv_sqrt.unsqueeze(0)  # [bs, L]
        norm_sq = norm_sq + (M_block**2).sum()

        # Compute M[j, i] for all (i, j) pairs in this block
        # i ranges over [i0:i1], j ranges over [0:L]
        row_idx = torch.arange(i0, i1, device=Q.device)  # [bs]
        col_idx = torch.arange(L, device=Q.device)  # [L]
        # For M^T contribution: M[j, i] for j in [0,L), i in [i0,i1)
        # Expand to compute batch of entries
        ii_exp = col_idx.unsqueeze(1).expand(L, bs).reshape(-1)  # [L*bs]
        jj_exp = row_idx.unsqueeze(0).expand(L, bs).reshape(-1)  # [L*bs]
        M_T_entries = get_M_entries_batch(
            Q, K, lse, d_k_inv_sqrt, scale, ii_exp, jj_exp
        )
        M_T_block = M_T_entries.reshape(L, bs).T  # [bs, L] — M[j,i] reshaped
        inner_sq = inner_sq + (M_block * M_T_block).sum()

    M_fro = torch.sqrt(norm_sq).item()
    asym_sq = (norm_sq - inner_sq) / 2.0
    asym_sq = asym_sq.clamp(min=0.0)
    G = (torch.sqrt(asym_sq) / (torch.sqrt(norm_sq) + EPSILON)).item()
    return G, M_fro


def compute_routing_features_materialized(
    M, rank, svd_method="randomized", target_cv=0.05, seed=42
):
    """Below-threshold all-in-one: materialize everything.

    Matches shade's _compute_routing_features semantics:
      - sigma2 = second singular value of M
      - phi_hat = 1 - sigma2 (spectral gap conductance)
      - commutator_norm = ||[M_sym, M_asym]||_F / (||M||_F + eps)
      - sigma2_asym = second singular value of M_asym (raw)
    """
    # If there's no second singular value, sigma2 = 0 and phi_hat = 1.0
    # (maximal spectral gap, i.e. rank-1 operator).
    sigma = torch.linalg.svdvals(M)
    sigma2 = sigma[1].item() if len(sigma) > 1 else 0.0
    phi_hat = 1.0 - sigma2

    G, M_fro = compute_G_materialized(M)

    M_sym = (M + M.T) / 2.0
    M_asym = (M - M.T) / 2.0
    sigma_asym = torch.linalg.svdvals(M_asym)
    sigma2_asym = sigma_asym[1].item() if len(sigma_asym) > 1 else 0.0

    # Commutator: [M_sym, M_asym] = M_sym @ M_asym - M_asym @ M_sym
    comm = M_sym @ M_asym - M_asym @ M_sym
    commutator_norm = torch.linalg.norm(comm, "fro").item() / (M_fro + EPSILON)

    # Curl
    C = estimate_curl_materialized(M, target_cv, seed)

    # Pythagorean
    Gamma = math.sqrt(max(G**2 - C**2, 0.0))

    curl_ratio = C / (G + EPSILON)

    return {
        "phi_hat": phi_hat,
        "sigma2": sigma2,
        "G": G,
        "Gamma": Gamma,
        "C": C,
        "curl_ratio": curl_ratio,
        "sigma2_asym": sigma2_asym,
        "commutator_norm": commutator_norm,
    }


def compute_routing_features_matrix_free(
    Q,
    K,
    d_k_inv_sqrt,
    scale,
    lse,
    rank,
    svd_method="randomized",
    block_size=256,
    target_cv=0.05,
    seed=42,
):
    """Above-threshold matrix-free routing features."""
    L = Q.shape[0]
    device = Q.device

    # We need at least 2 singular values to get sigma2
    k = min(max(rank, 2), L - 1)

    matvec = lambda v: matvec_M_blocked(Q, K, v, d_k_inv_sqrt, scale, block_size)
    matvec_t = lambda u: matvec_MT_blocked(Q, K, u, d_k_inv_sqrt, scale, block_size)

    if svd_method == "lanczos":
        _, S, _ = svd_via_lanczos(
            matvec, matvec_t, L, k, max(2 * k + 2, 20), str(device)
        )
    else:
        _, S, _ = randomized_svd(matvec, matvec_t, L, k, device=str(device))

    # If there's no second singular value, sigma2 = 0 and phi_hat = 1.0
    # (maximal spectral gap, i.e. rank-1 operator).
    S_sorted, _ = torch.sort(S, descending=True)
    sigma2 = S_sorted[1].item() if len(S_sorted) > 1 else 0.0
    phi_hat = 1.0 - sigma2

    M_fro_norm = compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size)
    M_fro_val = M_fro_norm.item()

    G, _ = compute_G_matrix_free(Q, K, d_k_inv_sqrt, scale, block_size)

    C = estimate_curl_matrix_free(
        Q, K, lse, d_k_inv_sqrt, scale, M_fro_val, target_cv, seed
    )

    Gamma = math.sqrt(max(G**2 - C**2, 0.0))
    curl_ratio = C / (G + EPSILON)

    return {
        "phi_hat": phi_hat,
        "sigma2": sigma2,
        "G": G,
        "Gamma": Gamma,
        "C": C,
        "curl_ratio": curl_ratio,
        "sigma2_asym": None,  # TODO: implement matrix-free sigma2_asym
        "commutator_norm": None,  # TODO: implement matrix-free commutator_norm
    }
