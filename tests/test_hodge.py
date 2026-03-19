import math

import torch

from glassbox.hodge import (
    adaptive_curl_samples,
    compute_G_materialized,
    compute_G_matrix_free,
    compute_routing_features_materialized,
    compute_routing_features_matrix_free,
    estimate_curl_materialized,
    estimate_curl_matrix_free,
    sample_triangles,
)
from glassbox.svd import (
    compute_degree_normalized_M,
    compute_dk_blocked,
    compute_logsumexp_blocked,
)


def _make_M(L, D, seed=42):
    """Helper: generate Q, K, scale, A, M and related quantities."""
    torch.manual_seed(seed)
    Q = torch.randn(L, D)
    K = torch.randn(L, D)
    scale = 1.0 / math.sqrt(D)
    A = torch.softmax(Q @ K.T * scale, dim=-1)
    M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
    return Q, K, scale, A, M, d_k_inv_sqrt


def test_sample_triangles():
    tri = sample_triangles(10, 20, seed=0)
    assert tri.shape[1] == 3
    assert len(tri) == 20
    # All i < j < k
    assert (tri[:, 0] < tri[:, 1]).all()
    assert (tri[:, 1] < tri[:, 2]).all()
    # No duplicates
    tri_set = set(map(tuple, tri))
    assert len(tri_set) == 20


def test_sample_triangles_small():
    """n < 3 should return empty."""
    tri = sample_triangles(2, 10)
    assert len(tri) == 0


def test_adaptive_curl_samples_small():
    assert adaptive_curl_samples(3) == 0
    assert adaptive_curl_samples(4) > 0


def test_curl_materialized_known_symmetric():
    """A symmetric M should have near-zero curl."""
    Q, K, scale, A, M, d_k_inv_sqrt = _make_M(16, 4)
    M_sym = (M + M.T) / 2.0
    C = estimate_curl_materialized(M_sym)
    assert C < 0.01, f"Symmetric matrix should have ~0 curl, got {C}"


def test_curl_materialized_vs_matrix_free():
    """Both paths should agree within tolerance."""
    Q, K, scale, A, M, d_k_inv_sqrt = _make_M(16, 4)
    lse = compute_logsumexp_blocked(Q, K, scale)
    M_fro = torch.linalg.norm(M, "fro").item()

    C_mat = estimate_curl_materialized(M, target_cv=0.05, seed=42)
    C_mf = estimate_curl_matrix_free(
        Q, K, lse, d_k_inv_sqrt, scale, M_fro, target_cv=0.05, seed=42
    )
    # They use the same triangle samples and same M entries, so should be close
    assert abs(C_mat - C_mf) < 0.05, f"Curl mismatch: mat={C_mat}, mf={C_mf}"


def test_G_materialized_vs_matrix_free():
    Q, K, scale, A, M, d_k_inv_sqrt = _make_M(16, 4)
    G_mat, fro_mat = compute_G_materialized(M)
    _, d_k_inv_sqrt_mf = compute_dk_blocked(Q, K, scale)
    G_mf, fro_mf = compute_G_matrix_free(Q, K, d_k_inv_sqrt_mf, scale, block_size=4)
    assert abs(G_mat - G_mf) < 0.05, f"G mismatch: mat={G_mat}, mf={G_mf}"
    assert abs(fro_mat - fro_mf) < 0.05, f"Fro mismatch: mat={fro_mat}, mf={fro_mf}"


def test_pythagorean_identity():
    """Verify G^2 >= C^2 and Gamma = sqrt(G^2 - C^2) is real and non-negative."""
    Q, K, scale, A, M, d_k_inv_sqrt = _make_M(20, 4, seed=99)
    features = compute_routing_features_materialized(M, rank=4)
    G = features["G"]
    C = features["C"]
    Gamma = features["Gamma"]
    # Pythagorean: G^2 = Gamma^2 + C^2 (approximately)
    assert abs(G**2 - Gamma**2 - C**2) < 0.01, (
        f"Pythagorean violation: G={G}, Gamma={Gamma}, C={C}"
    )


def test_routing_features_consistency():
    """All features should have expected keys and reasonable ranges."""
    Q, K, scale, A, M, d_k_inv_sqrt = _make_M(16, 4)
    features = compute_routing_features_materialized(M, rank=4)
    expected_keys = {
        "phi_hat",
        "sigma2",
        "G",
        "Gamma",
        "C",
        "curl_ratio",
        "sigma2_asym",
        "commutator_norm",
    }
    assert set(features.keys()) == expected_keys
    # sigma2 is the second singular value (raw), should be in [0, 1] for M
    assert 0.0 <= features["sigma2"] <= 1.0
    assert 0.0 <= features["phi_hat"] <= 1.0
    assert features["G"] >= 0.0
    assert features["C"] >= 0.0
    assert features["Gamma"] >= 0.0


def test_symmetric_matrix_zero_curl():
    """M = M^T should give C ~ 0, G ~ 0."""
    torch.manual_seed(77)
    # Create a symmetric positive matrix
    X = torch.randn(12, 12)
    M = X @ X.T
    M = M / M.sum(dim=1, keepdim=True)  # row-normalize
    M = (M + M.T) / 2.0  # symmetrize

    G, _ = compute_G_materialized(M)
    C = estimate_curl_materialized(M)
    assert G < 0.01, f"Symmetric M should have G~0, got {G}"
    assert C < 0.01, f"Symmetric M should have C~0, got {C}"


def test_routing_features_matrix_free_keys():
    """Matrix-free features should have null for sigma2_asym and commutator_norm."""
    Q, K, scale, A, M, d_k_inv_sqrt = _make_M(16, 4)
    _, d_k_inv_sqrt_mf = compute_dk_blocked(Q, K, scale)
    lse = compute_logsumexp_blocked(Q, K, scale)
    features = compute_routing_features_matrix_free(
        Q, K, d_k_inv_sqrt_mf, scale, lse, rank=4
    )
    assert features["sigma2_asym"] is None
    assert features["commutator_norm"] is None
    assert 0.0 <= features["sigma2"] <= 1.0
