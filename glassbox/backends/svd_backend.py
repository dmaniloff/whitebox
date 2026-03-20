"""
Custom vLLM attention backend that wraps Triton attention to perform
matrix-free SVD of attention matrices at configurable intervals.

Usage:
    1. Import this module (triggers @register_backend)
    2. Launch vLLM with attention_backend="CUSTOM", enforce_eager=True

Configuration via GlassboxConfig (see glassbox/config.py):
    - glassbox.yaml (primary)
    - Legacy GLASSBOX_SVD_* env vars (deprecated, auto-migrated)
    - Programmatic kwargs
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import IO

import torch
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from glassbox.config import GlassboxConfig
from glassbox.results import (
    DegreeNormalizedFeatures,
    SVDSnapshot,
    ScoresMatrixFeatures,
)
from glassbox.hodge import (
    compute_routing_features_materialized,
    compute_routing_features_matrix_free,
)
from glassbox.svd import (
    compute_degree_normalized_M,
    compute_dk_blocked,
    compute_logsumexp_blocked,
    matvec_M_blocked,
    matvec_MT_blocked,
    matvec_S,
    matvec_ST,
    randomized_svd,
    svd_via_lanczos,
)

logger = logging.getLogger(__name__)

# Matches layer index from various model naming conventions:
#   OPT/Qwen/Llama: "model.layers.0.self_attn"
#   GPT-2:          "transformer.h.0.attn.attn"
_LAYER_IDX_RE = re.compile(r"(?:layers|\.h)\.(\d+)")


@dataclass
class PerLayerSVDState:
    """Accumulates Q slices for a single attention layer."""

    q_buffer: list[torch.Tensor] = field(default_factory=list)
    step: int = 0


@dataclass
class ReqTracker:
    """Tracks request boundaries across layers during prefill.

    request_id is incremented once per new request when layer 0 sees a
    prefill (q_span > 1).  Previous versions used an in_prefill flag that
    was cleared during decode, but with max_tokens=1 there is no decode
    step so the flag was never cleared and request_id stopped incrementing.
    """

    request_id: int = -1


@register_backend(AttentionBackendEnum.CUSTOM)
class SVDTritonAttentionBackend(TritonAttentionBackend):
    """Drop-in replacement for TritonAttentionBackend that runs matrix-free
    SVD on the scores matrix during decode."""

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SVDTritonAttentionImpl"]:
        return SVDTritonAttentionImpl


class SVDTritonAttentionImpl(TritonAttentionImpl):
    """Wraps TritonAttentionImpl.forward() to accumulate Q and periodically
    extract K from the paged cache for matrix-free SVD."""

    # Class-level config; immutable after creation.
    # If you need to override, set it before vLLM creates the engine.
    # vLLM controls the constructor signature so we can't pass it in.
    config: GlassboxConfig = GlassboxConfig()

    # Class-level layer state; shared mutable.
    # Shared by all impl instances (one per layer or shared).
    # Keyed by layer_name (e.g. "model.layers.0.self_attn").
    state_dict: dict[str, PerLayerSVDState] = {}

    # Class-level request tracking; shared mutable.
    req_tracker: ReqTracker = ReqTracker()

    # Class-level output file handle; mutable.
    # Same rationale as other class-level state: vLLM may create one attention
    # impl per layer (many instances of SVDTritonAttentionImpl).
    # By keeping one handle on the class (_output_fh), every layer uses the same open file.
    _output_fh: IO | None = None

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Run normal Triton attention (unchanged)
        result = super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

        # 2. Skip during profiling
        if attn_metadata is None:
            return result

        # Skip if no signals enabled
        if not (self.config.scores_matrix.enabled or self.config.degree_normalized_matrix.enabled):
            return result

        # 3. Accumulate Q for the first sequence in the batch
        layer_name = getattr(layer, "layer_name", None)
        if layer_name is None:
            return result

        cls = type(self)
        state = cls.state_dict.get(layer_name)
        if state is None:
            state = PerLayerSVDState()
            cls.state_dict[layer_name] = state

        m = _LAYER_IDX_RE.search(layer_name)
        layer_idx: int | None = int(m.group(1)) if m else None

        # query shape: [num_tokens, num_heads, head_size]
        # query_start_loc is a cumulative offset tensor for sequences in the
        # batch.  For sequence 0 the Q rows live at
        #   query[query_start_loc[0] : query_start_loc[1]]
        # During prefill this span equals the prompt length; during decode
        # it is exactly 1.
        q_start = attn_metadata.query_start_loc[0].item()
        q_end = attn_metadata.query_start_loc[1].item()
        q_span = q_end - q_start

        if q_span > 1:  # prefill = new request
            # Increment request_id only on layer 0 to avoid double-counting
            # across the 12 layers that all see the same prefill batch.
            if layer_idx == 0:
                cls.req_tracker.request_id += 1
            state.q_buffer = []
            state.step = 0

        state.q_buffer.append(query[q_start:q_end].detach().clone())
        state.step += 1

        # 4. Check per-signal intervals and run SVD
        spectral_due = (
            self.config.scores_matrix.enabled
            and state.step % self.config.scores_matrix.interval == 0
        )
        normalized_due = (
            self.config.degree_normalized_matrix.enabled
            and state.step % self.config.degree_normalized_matrix.interval == 0
        )
        if spectral_due or normalized_due:
            try:
                self._run_svd(
                    layer_name, layer_idx, state, kv_cache, attn_metadata,
                    run_spectral=spectral_due, run_normalized=normalized_due,
                )
            except Exception:
                logger.exception(
                    "[SVD] error in layer %s at step %d", layer_name, state.step
                )

        return result

    @staticmethod
    def _extract_k_from_cache(
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        seq_idx: int = 0,
    ) -> torch.Tensor:
        """Extract the full K matrix for one sequence from the paged cache.

        Args:
            kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: contains block_table and seq_lens
            seq_idx: which sequence in the batch (default 0)

        Returns:
            K tensor of shape [seq_len, num_kv_heads, head_size]
        """
        # Unbind the KV dimension to get key_cache
        key_cache, _ = kv_cache.unbind(dim=1)
        # key_cache: [num_blocks, block_size, num_kv_heads, head_size]

        block_size = key_cache.shape[1]
        seq_len = attn_metadata.seq_lens[seq_idx].item()

        # Number of blocks needed for this sequence
        num_blocks_needed = (seq_len + block_size - 1) // block_size

        # Get physical block indices for this sequence
        block_indices = attn_metadata.block_table[seq_idx, :num_blocks_needed]

        # Gather the blocks: [num_blocks_needed, block_size, num_kv_heads, head_size]
        k_blocks = key_cache[block_indices]

        # Reshape to [num_blocks_needed * block_size, num_kv_heads, head_size]
        k_flat = k_blocks.reshape(-1, k_blocks.shape[2], k_blocks.shape[3])

        # Trim to actual sequence length
        k_flat = k_flat[:seq_len]

        # Cast to float if FP8
        return k_flat.float()

    def _run_svd(
        self,
        layer_name: str,
        layer_idx: int | None,
        state: PerLayerSVDState,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        run_spectral: bool = True,
        run_normalized: bool = False,
    ) -> None:
        # Stack accumulated Q: [L_q, num_heads, head_size]
        Q_all = torch.cat(state.q_buffer, dim=0).float()
        L_q = Q_all.shape[0]

        # Extract K: [L_k, num_kv_heads, head_size]
        K_all = self._extract_k_from_cache(kv_cache, attn_metadata, seq_idx=0)
        L_k = K_all.shape[0]

        # Q and K should match now that we capture prefill Q, but handle
        # edge cases (chunked prefill, sequence reordering) by aligning
        # from the end.
        L = min(L_q, L_k)
        if L < 2:
            return
        Q_all = Q_all[:L]
        K_all = K_all[:L]

        # Union of active heads for signals that are due
        heads: set[int] = set()
        if run_spectral:
            heads.update(self.config.scores_matrix.heads)
        if run_normalized:
            heads.update(self.config.degree_normalized_matrix.heads)

        for head_idx in sorted(heads):
            if head_idx >= Q_all.shape[1]:
                continue

            # Handle GQA: map head_idx to KV head
            kv_head_idx = head_idx // self.num_queries_per_kv

            Qh = Q_all[:, head_idx, :]  # [L, d]
            Kh = K_all[:, kv_head_idx, :]  # [L, d]

            if run_spectral and head_idx in self.config.scores_matrix.heads:
                self._run_svd_scores(layer_name, layer_idx, state, head_idx, Qh, Kh, L)
            if run_normalized and head_idx in self.config.degree_normalized_matrix.heads:
                self._run_svd_normalized(
                    layer_name, layer_idx, state, head_idx, Qh, Kh, L
                )

    def _run_svd_scores(
        self,
        layer_name: str,
        layer_idx: int | None,
        state: PerLayerSVDState,
        head_idx: int,
        Qh: torch.Tensor,
        Kh: torch.Tensor,
        L: int,
    ) -> None:
        """SVD of the scores matrix S = QK^T."""
        cfg = self.config.scores_matrix
        device = Qh.device
        matvec = lambda v, Q=Qh, K=Kh: matvec_S(Q, K, v)
        matvec_t = lambda u, Q=Qh, K=Kh: matvec_ST(Q, K, u)

        k = min(cfg.rank, L - 1)

        if cfg.method == "lanczos":
            _, S, _ = svd_via_lanczos(
                matvec=matvec,
                matvec_t=matvec_t,
                dim=L,
                k=k,
                iters=max(2 * k + 2, 20),
                device=str(device),
            )
        else:
            _, S, _ = randomized_svd(
                matvec=matvec,
                matvec_t=matvec_t,
                dim=L,
                k=k,
                device=str(device),
            )

        sv_list = S.cpu().tolist()
        snapshot = SVDSnapshot(
            feature_group="scores_matrix",
            request_id=type(self).req_tracker.request_id,
            layer=layer_name,
            layer_idx=layer_idx,
            head=head_idx,
            step=state.step,
            L=L,
            singular_values=sv_list,
            features=ScoresMatrixFeatures.from_singular_values(sv_list),
        )
        self._emit_result(snapshot)

    def _run_svd_normalized(
        self,
        layer_name: str,
        layer_idx: int | None,
        state: PerLayerSVDState,
        head_idx: int,
        Qh: torch.Tensor,
        Kh: torch.Tensor,
        L: int,
    ) -> None:
        """SVD of the degree-normalized cross-operator M."""
        cfg = self.config.degree_normalized_matrix
        scale = 1.0 / math.sqrt(Qh.shape[1])
        k = min(cfg.rank, L - 1)

        if L <= cfg.threshold:
            # TIER 1: materialize
            A = torch.softmax(Qh @ Kh.T * scale, dim=-1)
            M, _, d_k_inv_sqrt = compute_degree_normalized_M(A)
            sigma = torch.linalg.svdvals(M)
            sv_list = sigma[:k].cpu().tolist()
            tier = "materialized"
            hodge = (
                compute_routing_features_materialized(
                    M,
                    k,
                    cfg.method,
                    cfg.hodge_target_cv,
                    cfg.hodge_curl_seed,
                )
                if cfg.hodge
                else None
            )
        else:
            # TIER 2: matrix-free
            _, d_k_inv_sqrt = compute_dk_blocked(Qh, Kh, scale, cfg.block_size)
            device = Qh.device
            matvec = lambda v: matvec_M_blocked(
                Qh, Kh, v, d_k_inv_sqrt, scale, cfg.block_size
            )
            matvec_t = lambda u: matvec_MT_blocked(
                Qh, Kh, u, d_k_inv_sqrt, scale, cfg.block_size
            )

            if cfg.method == "lanczos":
                _, S, _ = svd_via_lanczos(
                    matvec, matvec_t, L, k, max(2 * k + 2, 20), str(device)
                )
            else:
                _, S, _ = randomized_svd(matvec, matvec_t, L, k, device=str(device))

            sv_list = S.cpu().tolist()
            tier = "matrix_free"

            if cfg.hodge:
                lse = compute_logsumexp_blocked(Qh, Kh, scale, cfg.block_size)
                hodge = compute_routing_features_matrix_free(
                    Qh,
                    Kh,
                    d_k_inv_sqrt,
                    scale,
                    lse,
                    k,
                    cfg.method,
                    cfg.block_size,
                    cfg.hodge_target_cv,
                    cfg.hodge_curl_seed,
                )
            else:
                hodge = None

        snapshot = SVDSnapshot(
            feature_group="degree_normalized_matrix",
            request_id=type(self).req_tracker.request_id,
            layer=layer_name,
            layer_idx=layer_idx,
            head=head_idx,
            step=state.step,
            L=L,
            singular_values=sv_list,
            tier=tier,
            features=DegreeNormalizedFeatures.from_singular_values(
                sv_list, routing=hodge if hodge else None,
            ),
        )
        self._emit_result(snapshot)

    def _emit_result(self, snapshot: SVDSnapshot) -> None:
        """Write SVD results to JSONL or log."""
        cls = type(self)

        if self.config.output:
            if cls._output_fh is None:
                cls._output_fh = open(self.config.output, "a")
            cls._output_fh.write(
                json.dumps(snapshot.model_dump(exclude_none=True)) + "\n"
            )
            cls._output_fh.flush()
        else:
            k = len(snapshot.singular_values)
            logger.info(
                "[SVD] %s head=%d step=%d L=%d top-%d singular values: %s",
                snapshot.layer,
                snapshot.head,
                snapshot.step,
                snapshot.L,
                k,
                snapshot.singular_values,
            )
