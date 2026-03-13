"""
Glassbox - A library for instrumenting and inspecting torch graph compilation.
"""

from .passes import custom_ops  # noqa: F401 - Import to register custom ops
from .passes import PostAttentionInjector, create_post_attention_injector

__all__ = ["PostAttentionInjector", "create_post_attention_injector"]
