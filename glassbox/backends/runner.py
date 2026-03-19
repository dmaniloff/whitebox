"""
Entry-point script that launches vLLM with the custom SVD attention backend.

Usage:
    python -m glassbox.backends.runner [OPTIONS]
    python -m glassbox.backends.runner --interval 16 --rank 2 --heads 0 1 2
    python -m glassbox.backends.runner --model facebook/opt-350m --method lanczos

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
    "--operator",
    type=click.Choice(["S", "M"]),
    default=None,
    help="Operator to SVD: S=scores, M=degree-normalized. [default: from config (S)]",
)
@click.option(
    "--threshold",
    type=int,
    default=None,
    help="Sequence length threshold for materialized vs matrix-free. [default: from config (2048)]",
)
@click.option(
    "--block-size",
    type=int,
    default=None,
    help="Block size for blocked-streaming matvecs. [default: from config (256)]",
)
@click.option(
    "--hodge/--no-hodge",
    default=None,
    help="Compute Hodge decomposition features. [default: from config (False)]",
)
@click.option(
    "--hodge-target-cv",
    type=float,
    default=None,
    help="Target CV for adaptive curl sampling. [default: from config (0.05)]",
)
@click.option(
    "--hodge-curl-seed",
    type=int,
    default=None,
    help="Seed for curl triangle sampling. [default: from config (42)]",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="JSONL output file path. [default: from config (log to stderr)]",
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
    output: str | None,
    operator: str | None,
    threshold: int | None,
    block_size: int | None,
    hodge: bool | None,
    hodge_target_cv: float | None,
    hodge_curl_seed: int | None,
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
    if output is not None:
        overrides["output"] = output
    if operator is not None:
        overrides["operator"] = operator
    if threshold is not None:
        overrides["threshold"] = threshold
    if block_size is not None:
        overrides["block_size"] = block_size
    if hodge is not None:
        overrides["hodge"] = hodge
    if hodge_target_cv is not None:
        overrides["hodge_target_cv"] = hodge_target_cv
    if hodge_curl_seed is not None:
        overrides["hodge_curl_seed"] = hodge_curl_seed

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
