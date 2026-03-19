"""
Explorations of matrix-free SVD algorithms.

A = softmax(QK^T / sqrt(d))
M = Dq_inv_sqrt * A * Dk_inv_sqrt
M is NOT symmetric in general. We compute SVD, not eigen-decomposition, using matrix–vector products with M and M^T
"""

import torch


def matvec_S(Q, K, v):
    """Calculate Sv = Q K^T v in two O(Ld) passes, avoid computing S: [L, L]."""
    # v: [L], Q,K: [L, d]
    z = K.T @ v  # [d]
    return Q @ z  # [L]


def matvec_ST(Q, K, u):
    """Calculate S^T u = K Q^T u in two O(Ld) passes, avoid computing S^T: [L, L]."""
    # u: [L], Q,K: [L, d]
    z = Q.T @ u  # [d]
    return K @ z  # [L]


def apply_A_blocked(Q, K, v, scale, block_size=256):
    """A @ v via blocked row-streaming. Peak memory: O(block_size * L_k)."""
    L_q = Q.shape[0]
    result = torch.zeros(L_q, device=Q.device, dtype=Q.dtype)
    for i0 in range(0, L_q, block_size):
        i1 = min(i0 + block_size, L_q)
        scores = Q[i0:i1] @ K.T * scale  # [bs, L_k]
        attn = torch.softmax(scores, dim=-1)  # [bs, L_k]
        result[i0:i1] = attn @ v  # [bs]
    return result


def apply_AT_blocked(Q, K, u, scale, block_size=256):
    """A^T @ u via blocked row-streaming."""
    L_k = K.shape[0]
    result = torch.zeros(L_k, device=K.device, dtype=K.dtype)
    for i0 in range(0, Q.shape[0], block_size):
        i1 = min(i0 + block_size, Q.shape[0])
        scores = Q[i0:i1] @ K.T * scale
        attn = torch.softmax(scores, dim=-1)  # [bs, L_k]
        result += attn.T @ u[i0:i1]  # [L_k]
    return result


def compute_dk_blocked(Q, K, scale, block_size=256, epsilon=1e-10):
    """Compute D_K (column sums of A) via apply_AT_blocked.

    Uses Moore-Penrose pseudoinverse: zero-degree positions get 0 instead of
    large values.
    """
    ones = torch.ones(Q.shape[0], device=Q.device, dtype=Q.dtype)
    d_k = apply_AT_blocked(Q, K, ones, scale, block_size)
    d_k_inv_sqrt = torch.where(
        d_k > epsilon, 1.0 / torch.sqrt(d_k), torch.zeros_like(d_k)
    )
    return d_k, d_k_inv_sqrt


def compute_logsumexp_blocked(Q, K, scale, block_size=256):
    """Precompute lse[i] = logsumexp(Q_i . K^T * scale) for all rows."""
    L_q = Q.shape[0]
    lse = torch.zeros(L_q, device=Q.device, dtype=Q.dtype)
    for i0 in range(0, L_q, block_size):
        i1 = min(i0 + block_size, L_q)
        scores = Q[i0:i1] @ K.T * scale  # [bs, L_k]
        lse[i0:i1] = torch.logsumexp(scores, dim=-1)
    return lse


def get_M_entries_batch(Q, K, lse, d_k_inv_sqrt, scale, ii, jj):
    """Compute M[ii, jj] on the fly. O(N*d) cost."""
    scores = (Q[ii] * K[jj]).sum(dim=-1) * scale  # [N]
    A_ij = torch.exp(scores - lse[ii])  # [N]
    return A_ij * d_k_inv_sqrt[jj]  # [N]


def matvec_M_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256):
    """M @ x = A @ (D_K^{-1/2} * x). D_Q^{-1/2} = I for row-stochastic A."""
    return apply_A_blocked(Q, K, d_k_inv_sqrt * x, scale, block_size)


def matvec_MT_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256):
    """M^T @ x = D_K^{-1/2} * (A^T @ x)."""
    return d_k_inv_sqrt * apply_AT_blocked(Q, K, x, scale, block_size)


def compute_M_fro_norm_blocked(Q, K, d_k_inv_sqrt, scale, block_size=256):
    """Compute ||M||_F without materializing M."""
    L_q = Q.shape[0]
    norm_sq = torch.tensor(0.0, device=Q.device, dtype=Q.dtype)
    for i0 in range(0, L_q, block_size):
        i1 = min(i0 + block_size, L_q)
        scores = Q[i0:i1] @ K.T * scale
        attn = torch.softmax(scores, dim=-1)  # [bs, L_k]
        M_block = attn * d_k_inv_sqrt.unsqueeze(0)  # broadcast [bs, L_k]
        norm_sq = norm_sq + (M_block**2).sum()
    return torch.sqrt(norm_sq)


def compute_degree_normalized_M(A, epsilon=1e-10):
    """
    Compute the degree-normalized cross-operator M from attention matrix A (materialized version).

    SHADE paper (Section 3.2.2, Equation 1): M = D_Q^{-1/2} @ A @ D_K^{-1/2}.
    M is a matrix whose structure reflects the pattern of information routing independent of degree heterogeneity,
    making spectral properties (singular values, asymmetry) comparable across heads and layers.

    Args:
        A: Attention matrix of shape (n_q, n_k)
        epsilon: Threshold below which degrees are treated as zero (default: 1e-10)

    Returns:
        M: Degree-normalized cross-operator of shape (n_q, n_k)
        d_q_inv_sqrt: Inverse sqrt of query degree vector of shape (n_q,)
        d_k_inv_sqrt: Inverse sqrt of key degree vector of shape (n_k,)
    """
    # Compute row sums (query degrees): d_Q_i = sum_j A_ij
    # shape: (n_q,); if A is softmax over rows, then D_Q = I
    d_q = A.sum(dim=1)

    # Compute column sums (key degrees): d_K_j = sum_i A_ij
    # shape: (n_k,)
    d_k = A.sum(dim=0)

    # Moore-Penrose pseudoinverse: zero out near-zero degrees
    d_q_inv_sqrt = torch.where(
        d_q > epsilon, 1.0 / torch.sqrt(d_q), torch.zeros_like(d_q)
    )
    d_k_inv_sqrt = torch.where(
        d_k > epsilon, 1.0 / torch.sqrt(d_k), torch.zeros_like(d_k)
    )

    M = (d_q_inv_sqrt[:, None] * A) * d_k_inv_sqrt[None, :]

    return M, d_q_inv_sqrt, d_k_inv_sqrt


def matvec_B_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size=256):
    """B @ x = M^T @ (M @ x)."""
    y = matvec_M_blocked(Q, K, x, d_k_inv_sqrt, scale, block_size)
    return matvec_MT_blocked(Q, K, y, d_k_inv_sqrt, scale, block_size)


def randomized_svd(matvec, matvec_t, dim, k, p=5, q=2, device="cuda"):
    """
    Matrix-free Randomized SVD for a (dim x dim) linear operator given matvecs.

    Computes a rank-k approximation M ≈ U diag(S) V^T without ever forming
    the full L×L matrix. The key insight is that for S = QK^T, we can compute
    Sv = Q(K^T v) and S^T u = K(Q^T u) in O(Ld) each, avoiding the O(L^2)
    cost of materializing S. The caller wraps this into the matvec / matvec_t
    callables, making this routine agnostic to the operator's internal structure.

    Algorithm (Halko, Martinsson, Tropp 2011):
      1. Draw a random Gaussian test matrix Ω of shape (dim, k+p).
      2. Form Y = M Ω via k+p matvec calls.
      3. (Optional) Run q power iterations for better spectral separation.
      4. Compute an orthonormal basis Q for range(Y).
      5. Project: B = Q^T M  (computed via matvec_t on columns of Q).
      6. SVD of the small (k+p)×dim matrix B, then lift U back.

    Args:
        matvec:   v -> M v,   callable on vectors of length `dim`.
        matvec_t: u -> M^T u, callable on vectors of length `dim`.
        dim: Ambient dimension (L, the sequence length).
        k:   Number of singular triplets to return.
        p:   Oversampling parameter (default 5).
        q:   Number of power iterations (default 2).
        device: Torch device.

    Returns:
        U: (dim, k) left singular vectors.
        S: (k,)    singular values (descending).
        V: (dim, k) right singular vectors.
    """
    # Clamp oversampling so k + p doesn't exceed dim
    p = min(p, max(dim - k, 0))

    # Step 1: random test matrix Ω
    Omega = torch.randn(dim, k + p, device=device)

    # Step 2: sample Y = M Ω
    Y = torch.stack([matvec(Omega[:, i]) for i in range(k + p)], dim=1)  # (dim, k+p)

    # Optional: power iterations to improve spectral separation.
    for _ in range(q):
        Z = torch.stack([matvec_t(Y[:, i]) for i in range(k + p)], dim=1)  # M^T Y
        Y = torch.stack([matvec(Z[:, i]) for i in range(k + p)], dim=1)  # M (M^T Y)

    # Step 3: orthonormal basis Q for range(Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")  # (dim, k+p)

    # Step 4: form small matrix B = Q^T M  (shape (k+p, dim))
    # We can compute B via B^T = M^T Q, using matvec_t.
    Bt = torch.stack([matvec_t(Q[:, i]) for i in range(k + p)], dim=1)  # (dim, k+p)
    B = Bt.T  # (k+p, dim)

    # Step 5: SVD of small B
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)

    # Step 6: lift left singular vectors back to original space
    U = Q @ U_hat[:, :k]  # (dim, k)
    V = Vt[:k, :].T  # (dim, k)
    return U, S[:k], V


def lanczos(operator, dim, k, iters, device):
    """
    operator: function v -> operator(v), expects v shape [dim]
    dim: dimension L
    k: number of Lanczos vectors kept (>= desired eigenvectors)
    iters: total Lanczos steps
    """
    Q = []
    alphas = []
    betas = []

    # start with random normalized vector
    q = torch.randn(dim, device=device)
    q = q / torch.linalg.norm(q)
    Q.append(q)

    beta = torch.tensor(0.0, device=device)

    for _ in range(iters):
        z = operator(Q[-1])  # apply B = Mᵀ M
        alpha = torch.dot(Q[-1], z)
        z = z - alpha * Q[-1] - beta * (Q[-2] if len(Q) > 1 else 0)

        # reorthogonalize for numerical stability
        for q_prev in Q:
            z -= torch.dot(z, q_prev) * q_prev

        beta = torch.linalg.norm(z)
        if beta < 1e-8:
            break

        Q.append(z / beta)
        alphas.append(alpha)
        betas.append(beta)

    # Build small tridiagonal matrix T
    T = torch.zeros(len(Q), len(Q), device=device)
    for i in range(len(Q)):
        if i < len(alphas):
            T[i, i] = alphas[i]
        if i < len(betas):
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

    # eigen-decomposition of T
    evals, evecs = torch.linalg.eigh(T)

    # reconstruct Ritz vectors
    V = torch.stack(Q, dim=1)  # [dim, m]
    ritz_vectors = V @ evecs  # [dim, m]

    return evals, ritz_vectors


def _principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Principal angles between column spaces of A and B (both (dim, k), assumed orthonormal-ish).
    Returns angles in radians, length k, sorted ascending.
    """
    # Orthonormalize for stability.
    QA, _ = torch.linalg.qr(A, mode="reduced")
    QB, _ = torch.linalg.qr(B, mode="reduced")
    # Singular values of QA^T QB are cosines of principal angles.
    s = torch.linalg.svdvals(QA.T @ QB).clamp(-1.0, 1.0)
    return torch.acos(s)


def svd_via_lanczos(matvec, matvec_t, dim: int, k: int, iters: int, device: str):
    """
    Compute top-k singular triplets using Lanczos on B = M^T M.

    Like randomized_svd, this never forms the L×L matrix. For S = QK^T the
    crucial observation is that Sv = Q(K^T v) costs only O(Ld) — two
    thin matmuls through the L×d factors — so each Lanczos step is O(Ld)
    rather than O(L^2).

    Lanczos builds a Krylov subspace {v, Bv, B^2 v, ...} for the symmetric
    operator B = M^T M using the supplied matvec / matvec_t pair. After
    `iters` steps it eigen-decomposes the resulting small tridiagonal matrix
    to obtain Ritz values λ_i ≈ σ_i^2 and Ritz vectors (right singular
    vectors). Left singular vectors are recovered via u_i = M v_i / σ_i.

    Args:
        matvec:   v -> M v,   callable on vectors of length `dim`.
        matvec_t: u -> M^T u, callable on vectors of length `dim`.
        dim:   Ambient dimension (L, the sequence length).
        k:     Number of singular triplets to return.
        iters: Number of Lanczos iterations.
        device: Torch device.

    Returns:
        U: (dim, k) left singular vectors.
        S: (k,)    singular values (descending).
        V: (dim, k) right singular vectors.
    """
    evals, ritz = lanczos(
        operator=lambda v: matvec_t(matvec(v)),
        dim=dim,
        k=max(2 * k, k + 2),
        iters=iters,
        device=device,
    )
    # torch.linalg.eigh returns ascending; take largest-k.
    idx = torch.argsort(evals, descending=True)[:k]
    lam = evals[idx].clamp(min=0.0)
    S = torch.sqrt(lam)
    V = ritz[:, idx]
    # U_i = (1/sigma_i) M v_i
    U_cols = []
    for i in range(k):
        if S[i] < 1e-12:
            U_cols.append(torch.zeros(dim, device=device, dtype=V.dtype))
        else:
            U_cols.append(matvec(V[:, i]) / S[i])
    U = torch.stack(U_cols, dim=1)
    return U, S, V


def compare_svd_results(matvec, matvec_t, U1, S1, V1, U2, S2, V2, trials: int = 8):
    """
    Compare two (U,S,V) factorizations for the same operator M using:
    - singular value agreement
    - principal angles between left/right subspaces
    - residual norms ||M v - s u|| and ||M^T u - s v||
    - randomized reconstruction check on random vectors
    """
    device = S1.device
    k = min(S1.numel(), S2.numel())
    S1 = S1[:k]
    S2 = S2[:k]
    U1 = U1[:, :k]
    U2 = U2[:, :k]
    V1 = V1[:, :k]
    V2 = V2[:, :k]

    # singular values (order them descending before comparing)
    S1s, _ = torch.sort(S1, descending=True)
    S2s, _ = torch.sort(S2, descending=True)
    sv_abs = (S1s - S2s).abs()
    sv_rel = sv_abs / torch.clamp(torch.max(S1s.abs(), S2s.abs()), min=1e-12)

    # subspace alignment
    ang_U = _principal_angles(U1, U2)
    ang_V = _principal_angles(V1, V2)

    # residuals
    mv_res_list: list[torch.Tensor] = []
    mtu_res_list: list[torch.Tensor] = []
    for i in range(k):
        s = torch.max(S1s[i], S2s[i]).clamp(min=1e-12)
        mv_res_list.append(torch.linalg.norm(matvec(V2[:, i]) - S2s[i] * U2[:, i]) / s)
        mtu_res_list.append(
            torch.linalg.norm(matvec_t(U2[:, i]) - S2s[i] * V2[:, i]) / s
        )
    mv_res: torch.Tensor = torch.stack(mv_res_list)
    mtu_res: torch.Tensor = torch.stack(mtu_res_list)

    # reconstruction check on random vectors: compare Mx to U diag(S) V^T x
    recon_list: list[torch.Tensor] = []
    for _ in range(trials):
        x = torch.randn(V1.shape[0], device=device)
        y = matvec(x)
        y1 = U1 @ (S1 * (V1.T @ x))
        y2 = U2 @ (S2 * (V2.T @ x))
        denom = torch.linalg.norm(y).clamp(min=1e-12)
        recon_list.append(
            torch.stack(
                [
                    torch.linalg.norm(y - y1) / denom,
                    torch.linalg.norm(y - y2) / denom,
                    torch.linalg.norm(y1 - y2) / denom,
                ]
            )
        )
    recon: torch.Tensor = torch.stack(recon_list, dim=0)  # (trials, 3)

    return {
        "k": k,
        "sv_abs_max": sv_abs.max().item(),
        "sv_rel_max": sv_rel.max().item(),
        "ang_U_max_deg": (ang_U.max() * 180.0 / torch.pi).item(),
        "ang_V_max_deg": (ang_V.max() * 180.0 / torch.pi).item(),
        "mv_res_max": mv_res.max().item(),
        "mtu_res_max": mtu_res.max().item(),
        "recon_M_minus_USVt_mean": recon[:, 0].mean().item(),
        "recon_M_minus_USVt_max": recon[:, 0].max().item(),
        "recon_method_diff_mean": recon[:, 2].mean().item(),
        "recon_method_diff_max": recon[:, 2].max().item(),
    }
