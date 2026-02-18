"""
Entry-point script that launches vLLM with the custom SVD attention backend.

Usage:
    python -m glassbox.svd_backend_runner [OPTIONS]
    python -m glassbox.svd_backend_runner --interval 16 --rank 2 --heads 0 1 2
    python -m glassbox.svd_backend_runner --model facebook/opt-350m --method lanczos

Options can also be set via environment variables (prefix GLASSBOX_SVD_)
or a .env file. CLI args take precedence.
"""

from __future__ import annotations

import logging

import click
import vllm

# Import triggers @register_backend(AttentionBackendEnum.CUSTOM)
import glassbox.backends.svd_backend as svd_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    default="facebook/opt-125m",
    show_default=True,
    help="HuggingFace model name.",
)
@click.option(
    "--interval",
    type=int,
    default=None,
    help="Run SVD every N decode steps. [default: from config (32)]",
)
@click.option(
    "--rank",
    type=int,
    default=None,
    help="Number of singular values to compute. [default: from config (4)]",
)
@click.option(
    "--method",
    type=click.Choice(["randomized", "lanczos"]),
    default=None,
    help="SVD algorithm. [default: from config (randomized)]",
)
@click.option(
    "--heads",
    type=int,
    multiple=True,
    help="Head indices to analyze (repeatable). [default: from config ([0])]",
)
@click.option(
    "--max-tokens",
    type=int,
    default=64,
    show_default=True,
    help="Maximum tokens to generate.",
)
@click.option(
    "--prompt",
    default="The future of artificial intelligence is",
    show_default=True,
    help="Input prompt.",
)
def main(
    model: str,
    interval: int | None,
    rank: int | None,
    method: str | None,
    heads: tuple[int, ...],
    max_tokens: int,
    prompt: str,
) -> None:
    """Launch vLLM with the custom SVD attention backend."""

    # Build config: start from env vars / .env defaults, override with CLI args
    overrides: dict = {}
    if interval is not None:
        overrides["interval"] = interval
    if rank is not None:
        overrides["rank"] = rank
    if method is not None:
        overrides["method"] = method
    if heads:
        overrides["heads"] = list(heads)

    # vLLM calls impl_cls(). There doesn't seem to be a way to inject extra args through the vLLM call path.
    # So we set the config as a class variable on SVDTritonAttentionImpl before vLLM creates the engine.
    config = svd_mod.SVDConfig(**overrides)
    svd_mod.SVDTritonAttentionImpl.config = config

    logger.info("Creating vLLM engine with CUSTOM attention backend")
    logger.info("Model: %s", model)
    logger.info(
        "SVD config: interval=%s rank=%s method=%s heads=%s",
        config.interval,
        config.rank,
        config.method,
        config.heads,
    )

    llm = vllm.LLM(
        model=model,
        attention_backend="CUSTOM",
        enforce_eager=True,
    )

    logger.info("Starting generation...")
    outputs = llm.generate(
        [prompt],
        vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens),
    )

    for output in outputs:
        logger.info("Prompt: %s", output.prompt)
        logger.info("Generated: %s", output.outputs[0].text)


if __name__ == "__main__":
    main()
