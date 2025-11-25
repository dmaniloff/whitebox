# glassbox


## What It Does

Registers a custom [PyTorch FX](https://docs.pytorch.org/docs/stable/fx.html) inductor pass that inserts monitoring passthrough nodes that compute and print attention means.

## How to Run

```bash
python -m glassbox.main
```

The script will:
- Load the `facebook/opt-125m` model with vLLM
- Apply the custom pass during compilation
- Print attention means for each layer at each generation step
- Write the graph before and after the pass to `graph_before.txt` and `graph_after.txt` respectively

## Demo
Check out `demo/sample_run.txt` for a sample run and the corresponding `graph_before.txt` and `graph_after.txt` files.

## Implementation Details

The pass is registered via vLLM's [`CompilationConfig`](https://docs.vllm.ai/en/latest/api/vllm/config/#vllm.config.CompilationConfig) using the `inductor_compile_config` parameter with a custom `post_grad_custom_post_pass`.

See `glassbox/passes/mean.py` for the FX graph transformation logic.

## Dev setup
There's no pyproject.toml file yet because I set this up manually for now:

- Spin up a Ubuntu 24.04 Deep Learning AMI. Use g5.xlarge which gives you a NVIDIA A10G GPU.
- This comes with a Pytorch env, to actvate it,run: 'source /opt/pytorch/bin/activate '.
- `pip install vllm`.
