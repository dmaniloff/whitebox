"""
Custom vLLM attention backend that wraps Triton attention to perform
matrix-free SVD of the scores matrix S = Q K^T at configurable intervals.

Usage:
    1. Import this module (triggers @register_backend)
    2. Launch vLLM with attention_backend="CUSTOM", enforce_eager=True

Configuration via env vars or .env file (see SVDConfig):
    GLASSBOX_SVD_INTERVAL  - run SVD every N decode steps (default: 32)
    GLASSBOX_SVD_RANK      - number of singular values to compute (default: 4)
    GLASSBOX_SVD_METHOD    - "randomized" or "lanczos" (default: "randomized")
    GLASSBOX_SVD_HEADS     - JSON list of head indices (default: '[0]', e.g. '[0,1,2]')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from glassbox.svd import matvec_S, matvec_ST, randomized_svd, svd_via_lanczos

logger = logging.getLogger(__name__)


class SVDConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GLASSBOX_SVD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    interval: int = 32
    rank: int = 4
    method: Literal["randomized", "lanczos"] = "randomized"
    heads: list[int] = [0]  # JSON list in env, e.g. GLASSBOX_SVD_HEADS='[0,1,2]'


@dataclass
class PerLayerSVDState:
    """Accumulates Q slices for a single attention layer."""

    q_buffer: list[torch.Tensor] = field(default_factory=list)
    step: int = 0


# Module-level dict keyed by layer_name (e.g. "model.layers.0.self_attn")
_svd_state: dict[str, PerLayerSVDState] = {}


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

    # Class-level config; set before vLLM creates the engine.
    # vLLM controls the constructor signature so we can't pass it there.
    config: SVDConfig = SVDConfig()

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

        # 3. Accumulate Q for the first sequence in the batch
        layer_name = getattr(layer, "layer_name", None)
        if layer_name is None:
            return result

        state = _svd_state.get(layer_name)
        if state is None:
            state = PerLayerSVDState()
            _svd_state[layer_name] = state

        # query shape: [num_tokens, num_heads, head_size]
        # query_start_loc is a cumulative offset tensor for sequences in the
        # batch.  For sequence 0 the Q rows live at
        #   query[query_start_loc[0] : query_start_loc[1]]
        # During prefill this span equals the prompt length; during decode
        # it is exactly 1.
        q_start = attn_metadata.query_start_loc[0].item()
        q_end = attn_metadata.query_start_loc[1].item()
        state.q_buffer.append(query[q_start:q_end].detach().clone())
        state.step += 1

        # 4. Every self.config.interval steps, extract K and run SVD
        if state.step % self.config.interval == 0:
            try:
                self._run_svd(layer_name, state, kv_cache, attn_metadata)
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
        state: PerLayerSVDState,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
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

        for head_idx in self.config.heads:
            if head_idx >= Q_all.shape[1]:
                continue

            # Handle GQA: map head_idx to KV head
            kv_head_idx = head_idx // self.num_queries_per_kv

            Qh = Q_all[:, head_idx, :]  # [L, d]
            Kh = K_all[:, kv_head_idx, :]  # [L, d]

            device = Qh.device
            matvec = lambda v, Q=Qh, K=Kh: matvec_S(Q, K, v)
            matvec_t = lambda u, Q=Qh, K=Kh: matvec_ST(Q, K, u)

            k = min(self.config.rank, L - 1)

            if self.config.method == "lanczos":
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
            logger.info(
                "[SVD] %s head=%d step=%d L=%d top-%d singular values: %s",
                layer_name,
                head_idx,
                state.step,
                L,
                k,
                sv_list,
            )
