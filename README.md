# glassbox

Grab vLLM's attention.

## Dev setup

- I spun up a Ubuntu 24.04 Deep Learning AMI (e.g. g5.xlarge for NVIDIA A10G).
- Activate the PyTorch env: `source /opt/pytorch/bin/activate`
- `pip install vllm==0.15.1`

## Approaches

### 1. Custom attention backend (`backends/`)

A custom vLLM attention backend that wraps the Triton backend, accumulates Q tokens during decode, extracts K from the paged KV cache, and runs matrix-free SVD at configurable intervals — all without modifying vLLM source.

See `glassbox/backends/svd_backend.py` for the implementation and `glassbox/svd.py` for the SVD algorithms (randomized SVD and Lanczos).

#### Run

```bash
GLASSBOX_SVD_INTERVAL=16 GLASSBOX_SVD_RANK=2 \
  python -m glassbox.backends.runner
```

#### Configuration

Configuration is managed by the `SVDConfig` pydantic-settings model in `glassbox/backends/svd_backend.py`. Values can be set via environment variables or a `.env` file.

| Environment variable | Description | Default |
|---|---|---|
| `GLASSBOX_SVD_INTERVAL` | Run SVD every N decode steps | `32` |
| `GLASSBOX_SVD_RANK` | Number of singular values to compute | `4` |
| `GLASSBOX_SVD_METHOD` | `"randomized"` or `"lanczos"` | `"randomized"` |
| `GLASSBOX_SVD_HEADS` | JSON list of head indices, e.g. `'[0,1,2]'` | `[0]` |
| `GLASSBOX_MODEL` | HuggingFace model name (runner only) | `"facebook/opt-125m"` |

#### How it works

1. `SVDTritonAttentionBackend` registers itself as the `CUSTOM` backend via vLLM's `@register_backend` decorator.
2. `SVDTritonAttentionImpl.forward()` calls the normal Triton attention kernel, then accumulates Q tensors for the first sequence in the batch.
3. Every `GLASSBOX_SVD_INTERVAL` decode steps, it extracts K from the paged KV cache and computes the top singular values of `S = Q K^T` using matrix-free methods (never materializing the full L×L matrix).

### 2. FX graph instrumentation (torch.compile pass -- this is NOT working yet) (`passes/`)

Registers a custom [PyTorch FX](https://docs.pytorch.org/docs/stable/fx.html) inductor pass that inserts monitoring passthrough nodes to compute and print attention means.

See `glassbox/passes/injector.py` for the FX graph transformation logic.

#### Run

```bash
python -m glassbox.main
```

#### Demo
Check out `demo/sample_run.txt` for a sample run and the corresponding `graph_before.txt` and `graph_after.txt` files.

