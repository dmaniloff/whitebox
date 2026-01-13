"""Glassbox: Graph instrumentation for vLLM attention analysis.

Usage:
    glassbox [--injector=<type>] [--operation=<op>]
    glassbox -h | --help

Options:
    -h --help           Show this help message.
    -i --injector=<type>  Injector type: "post" or "before" [default: post].
    -o --operation=<op>   Operation type: "mean" or "qkv" [default: mean].

Notes:
    - "post" injector with "mean" op: captures attention output mean values
    - "before" injector with "qkv" op: captures Q, K, V tensors before attention
    - Other combinations may fail due to signature mismatches
"""

import huggingface_hub as hf
import torch
from docopt import docopt
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

from . import custom_ops  # noqa: F401 - Register custom ops
from .config import config
from .passes import BeforeAttentionInjector, PostAttentionInjector

INJECTORS = {
    "post": PostAttentionInjector,
    "before": BeforeAttentionInjector,
}

OPERATIONS = {
    "mean": torch.ops.glassbox.capture_mean.default,
    "qkv": torch.ops.glassbox.capture_qkv.default,
    "svd_lanczos": torch.ops.glassbox.svd_of_scores_matrix_lanczos.default,
    "svd_rnd": torch.ops.glassbox.svd_of_scores_matrix_rnd.default,
}


def main():
    args = docopt(__doc__)

    injector_type = args["--injector"]
    operation_type = args["--operation"]

    if (injector_cls := INJECTORS.get(injector_type)) is None:
        raise ValueError(
            f"Unknown injector: {injector_type}. Choose from: {list(INJECTORS.keys())}"
        )
    if (custom_op := OPERATIONS.get(operation_type)) is None:
        raise ValueError(
            f"Unknown operation: {operation_type}. Choose from: {list(OPERATIONS.keys())}"
        )

    print(f"Using injector: {injector_type}, operation: {operation_type}")

    compilation_config = CompilationConfig(
        splitting_ops=[],
        cudagraph_mode="NONE",
        inductor_compile_config={"post_grad_custom_post_pass": injector_cls(custom_op)},
    )

    hf.login(token=config.hf_token.get_secret_value())
    llm = LLM(model=config.model, compilation_config=compilation_config)

    prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")


if __name__ == "__main__":
    main()
