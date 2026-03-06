"""vLLM general plugin that registers the glassbox SVD attention backend.

This is loaded automatically by vLLM in all processes (API server, engine core,
workers) via the ``vllm.general_plugins`` entry point defined in pyproject.toml.
"""


def register_svd_backend():
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "glassbox.backends.svd_backend.SVDTritonAttentionBackend",
    )
