import huggingface_hub as hf
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

from .config import config
from .passes.mean import AttentionMeanPass


def main():
    compilation_config = CompilationConfig(
        splitting_ops=[],
        cudagraph_mode="NONE",
        # this didn't work
        # inductor_passes={
        #     # make sure this is exposed in __init__.py
        #     "attention_mean_pass": "glassbox.AttentionMeanPass"
        # }
        inductor_compile_config={"post_grad_custom_post_pass": AttentionMeanPass()},
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
