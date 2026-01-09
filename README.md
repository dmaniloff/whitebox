# glassbox

Grab vLLM's attention.

## By manipulating the graph produced by torch.compile's frontend

See https://blog.vllm.ai/2025/08/20/torch-compile.html

### How 

Registers a custom [PyTorch FX](https://docs.pytorch.org/docs/stable/fx.html) inductor pass that inserts monitoring passthrough nodes that compute and print attention means.

### Run in terminal

```bash
python -m glassbox.main
```

The script will:
- Load the `facebook/opt-125m` model with vLLM
- Apply the custom pass during compilation
- Print attention means for each layer at each generation step
- Write the graph before and after the pass to `graph_before.txt` and `graph_after.txt` respectively

### Demo
Check out `demo/sample_run.txt` for a sample run and the corresponding `graph_before.txt` and `graph_after.txt` files.

### Implementation Details

The pass is registered via vLLM's [`CompilationConfig`](https://docs.vllm.ai/en/latest/api/vllm/config/#vllm.config.CompilationConfig) using the `inductor_compile_config` parameter with a custom `post_grad_custom_post_pass`.

See `glassbox/passes/injector.py` for the FX graph transformation logic.

## By hooking into vllm/attention/layer.py inside Attention.forward
Three places where we might want attention matrices:
- Every generated token
- Every n tokens (windowed)
- Only once at the end of generation

In all cases, we want to avoid the full `L × L` matrix for all tokens. Ways to mitigate this:
- Only last‑row attention
- Snapshot a small window  `W × L` 
- Take a full `L × L` snapshot but infrequently.

And we can combine these strategieswith sampling only for certain tokens and/or layers.

Approach:
Add a “safety metadata” flag. Extend ForwardContext or AttentionMetadata with fields like:

```python
class SafetyConfig:
    enabled: bool
    snapshot_step: int  # e.g. token index in decode
    layers_to_dump: Optional[List[int]] = None  # default: all

# accessible via get_forward_context().safety_config for the request
```

When you submit the request to vLLM’s engine, add this safety config (e.g., via a custom field or by subclassing the request object).

Then, Hook in Attention.forward. After you reshape query:

```python
query = query.view(-1, self.num_heads, self.head_size)
# ... same for key/value
```

Insert: 

```python
ctx = get_forward_context()
safety_cfg = getattr(ctx, "safety_config", None)

if safety_cfg and safety_cfg.enabled:
    # step_idx must come from attn_metadata / ctx
    step_idx = ctx.attn_metadata.decode_position  # pseudo-code

    if step_idx == safety_cfg.snapshot_step:
        # clone Q for this layer / batch
        self._safety_q_buffer = query.detach().clone()
        # maybe also stash a reference to kv_cache or metadata
        self._safety_attn_metadata = ctx.attn_metadata
```

After the forward pass, compute attention rows

Two options:

Option 1: pure PyTorch (simpler, slower)
For the sequences/layers you care about:
Use self._safety_attn_metadata and KV cache manager to gather a dense K for each sequence (e.g. back to [L, H, d]).
Compute:

```python
# Q: [num_new_tokens, H, d], typically 1 × H × d in decode
# K_all: [L, H, d]
attn_logits = torch.einsum("qhd,lhd->qhl", Q, K_all) / math.sqrt(d)
attn_logits += causal_mask + any KV-specific masks
attn = attn_logits.softmax(dim=-1)  # [q, H, L]
```

Save attn for your safety logic.


Option 2: call vLLM’s paged attention op

Instead of reconstructing dense K:
- Use the block_tables, context_lens, kv_cache from attn_metadata and self.kv_cache (the paged KV blocks). 
intel.github.io
+1
- Call vllm._C.ops.paged_attention_vX (or the v1/v2 variant you use) with:
  - Q = saved self._safety_q_buffer.
  - The same K/V cache + block tables.
  - But in a mode where you only want softmax(QK^T), not the V multiply (this may require a tiny kernel modification or a wrapper).

This keeps you on the same optimized code path and lets you handle very long sequences efficiently.

## By writing a new attention backend (future work)

## Dev setup notes
There's no pyproject.toml file yet because I set this up manually for now:

- Spin up a Ubuntu 24.04 Deep Learning AMI. Use g5.xlarge which gives you a NVIDIA A10G GPU.
- This comes with a Pytorch env, to actvate it,run: 'source /opt/pytorch/bin/activate '.
- `pip install vllm`.
