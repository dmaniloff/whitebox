# glassbox

*Grab vLLM's attention.*

`glassbox` is a vLLM plugin for extracting model internals during inference and turning them into compact, structured signals for downstream reliability systems. These signals include spectral features and flow-based features that provide a routing-oriented view of model behavior.

The main use case is online or offline analysis of failure modes in LLM generation: hallucination detection, task drift detection, uncertainty quantification, routing analysis, and other forms of model-behavior monitoring.

The primary implementation is a custom vLLM attention backend in `glassbox/backends/`. There is also an experimental `torch.compile` / FX instrumentation path in `glassbox/passes/`, but the custom backend is the working path.

## What It Extracts

At configurable intervals during inference, `glassbox` can compute features from two operators:

1. The pre-softmax scores matrix `S = QK^T`
2. The degree-normalized post-softmax operator `M = D_Q^{-1/2} A D_K^{-1/2}`, where `A = softmax(QK^T / sqrt(d))`

For each tracked `(request, layer, head, step)`, it emits a JSONL record with:

- request metadata
- layer and head identifiers
- sequence length `L`
- top singular values
- derived spectral features
- optional routing / Hodge-style features for the normalized operator

Those snapshots are represented by `SVDSnapshot` in `glassbox/results.py`.

## Why This Exists

Transformer internals can reveal a great deal about model behavior. They are useful for monitoring, debugging, and failure analysis, but most of the underlying objects are too large to inspect directly in a practical inference setting.

Raw activations and full attention matrices are expensive to retain, and modern attention systems are specifically engineered to avoid materializing the full `L x L` object during efficient inference. Because of that, many tools for inspecting transformer internals live in research harnesses around HuggingFace models rather than in production-grade inference stacks.

That creates a gap between interpretability results in papers and practical deployment in real systems. `glassbox` is built to close that gap by efficiently extracting compact signals of how attention is behaving:

- Is one mode dominating, or are multiple modes active?
- Is routing bottlenecked through a narrow channel?
- Is behavior becoming more asymmetric or circulatory over time?
- Do certain layers or heads shift sharply when the model starts to drift or hallucinate?


## How It Integrates With vLLM

The package registers itself through vLLM's plugin entrypoint and exposes a `CUSTOM` attention backend:

- Entry point: `glassbox.vllm_plugin:register_svd_backend`
- Backend: `glassbox.backends.svd_backend.SVDTritonAttentionBackend`

At runtime, the backend:

1. Calls the normal Triton attention implementation unchanged.
2. Captures and accumulates `Q` slices across prefill and decode for the active sequence.
3. Extracts `K` from vLLM's paged KV cache when a snapshot is due.
4. Runs matrix-free SVD and optional routing analysis.
5. Emits JSONL rows with feature snapshots.

This lets you observe attention structure during real generation rather than in a separate offline re-run.

## How We Avoid Materializing The Full `L x L` Matrix

The key design goal is to avoid building full score or attention matrices whenever sequence length grows.

### 1. Matrix-free multiplies for `S = QK^T`

For any vector `v`:

- `Sv = Q(K^T v)`
- `S^T u = K(Q^T u)`

That means applying `S` or `S^T` only requires two thin `L x d` multiplies instead of constructing an `L x L` matrix. In code, this is implemented by:

- `matvec_S()` in `glassbox/svd.py`
- `matvec_ST()` in `glassbox/svd.py`

Both the randomized SVD and Lanczos implementations consume only these matvecs.

### 2. Randomized SVD and Lanczos operate on operators, not matrices

`glassbox/svd.py` provides:

- `randomized_svd()`
- `svd_via_lanczos()`

Both operate on callables `matvec` and `matvec_t`, so they never require the full matrix to exist in memory.

### 3. Blocked row streaming for post-softmax attention

For the degree-normalized operator, some quantities depend on softmaxed attention. When sequence length is large, `glassbox` uses blocked streaming:

- `apply_A_blocked()`
- `apply_AT_blocked()`
- `compute_dk_blocked()`
- `compute_logsumexp_blocked()`
- `matvec_M_blocked()`
- `matvec_MT_blocked()`

These compute the effect of `A` or `M` in row blocks, keeping memory bounded by the block size instead of `L^2`.

### 4. Two-tier execution for the normalized operator

For shorter sequences, `glassbox` will materialize the normalized operator because it is simple and exact. For longer sequences, it switches to the matrix-free path.

In practice:

- if `L <= threshold`, use a materialized path
- if `L > threshold`, use blocked matrix-free operators

That behavior is implemented in `_run_svd_normalized()` in `glassbox/backends/svd_backend.py`.

### 5. On-demand entry lookup for curl estimates

Some routing metrics need access to selected entries of `M`. Instead of building all of `M`, the matrix-free path computes only sampled entries on demand with `get_M_entries_batch()` in `glassbox/svd.py`.

## Features We Compute

### Scores matrix features (pre-softmax scores)

These come from the singular values of the pre-softmax scores matrix `S = QK^T`.

| Feature | Formula | Meaning |
|---|---|---|
| `sv1` | `σ₁(S)` | Leading singular value of the scores matrix. Strength of the dominant attention mode |
| `sv_ratio` | `σ₁(S) / σ₂(S)` | Spectral sharpness. High values suggest near-rank-1 structure; low values suggest multiple competing modes |
| `sv_entropy` | `-Σ pᵢ log pᵢ`, with `pᵢ = σᵢ / Σⱼ σⱼ` | Entropy of the normalized singular-value distribution. Measures how concentrated or diffuse the spectrum is |

### Routing features from the Degree-normalized matrix (post-softmax attention)

These come from the singular values and routing decomposition of the normalized operator `M`.

| Feature | Formula | Meaning |
|---|---|---|
| `sv1` | `σ₁(M)` | Leading singular value of `M`. Dominance of the top routing mode |
| `sv_ratio` | `σ₁(M) / σ₂(M)` | Separation between the top routing mode and the rest |
| `sv_entropy` | `-Σ pᵢ log pᵢ`, with `pᵢ = σᵢ / Σⱼ σⱼ` | Entropy of the normalized singular-value distribution. Spread of routing mass across modes |
| `sigma2` | `σ₂(M)` | Second singular value of `M`. Raw spectral-gap measure and persistence of non-dominant routing structure |
| `phi_hat` | `1 - σ₂(M)` | Conductance-like bottleneck score. High `φ̂` means attention concentrates through a single dominant mode; low `φ̂` means multiple competing routing paths |
| `G` | `‖M_asym‖_F / ‖M‖_F` | Total asymmetry. Fraction of `M`'s energy in the antisymmetric part, where `M_asym = (M - Mᵀ) / 2` |
| `Gamma` | `√(G² - C²)` | Gradient coefficient. The portion of asymmetry that is potential-driven rather than circulatory |
| `C` | `curl_RMS / (√2 · ‖M‖_F)` | Curl coefficient. The portion of asymmetry due to irreversible circulation, estimated by triangle sampling in the matrix-free path |
| `curl_ratio` | `C / (G + ε)` | Curl fraction. What share of total asymmetry is circulatory versus gradient-driven |
| `sigma2_asym` | `σ₂(M_asym)` | Second singular value of the antisymmetric part. Captures whether the irreversible component has multiple significant modes |
| `commutator_norm` | `‖[M_sym, M_asym]‖_F / ‖M‖_F` | Commutator norm. Measures how much the symmetric and antisymmetric parts interfere with each other, where `[A, B] = AB - BA` |

The routing and Hodge-style metrics live in `glassbox/hodge.py`. The feature schemas are defined in `glassbox/results.py`.

NOTE: `sigma2_asym` and `commutator_norm` are not yet implemented for the matrix-free normalized-operator path.


### More features coming soon

#### Transport features from the Degree-normalized matrix (post-softmax attention, value-weighted routing)

These move beyond pure attention geometry and start incorporating what is actually being transported through the head. 

#### Curl spectrum features from the Degree-normalized matrix (post-softmax attention, per-dimension value analysis)

These summarize how curl-like behavior is distributed across important value dimensions. 

#### LayerNorm-weighted features

These modulate routing features by the effective LayerNorm gain, with the goal of emphasizing heads and layers whose routed signal is more strongly amplified by the surrounding network.

## Output Format

The backend emits one JSON object per observation. Each row contains:

- `feature_group`: `scores_matrix` or `degree_normalized_matrix`
- `request_id`
- `layer` and `layer_idx`
- `head`
- `step`
- `L`
- `singular_values`
- `tier`: `materialized` or `matrix_free` for normalized-operator runs
- `features`: derived metrics for that observation

This format is designed to feed downstream systems directly. You can:

- train hallucination detectors on snapshot features
- compare truthful vs hallucinated generations
- monitor drift across prompts or tasks
- aggregate by head, layer, request, or dataset

The dataset extraction pipeline in `experiments/extract.py` can also write a wide Parquet file.

## Example Downstream Uses

- Hallucination detection: compare spectral and routing signatures between factual and hallucinated responses
- Task drift detection: identify layer/head regimes that diverge when the model loses the task
- Uncertainty quantification: use spectral concentration and routing asymmetry as auxiliary confidence signals
- Failure mode analysis: inspect how internal structure changes across prompts, models, or checkpoints

## Installation

This repository expects vLLM to be installed separately.

```bash
# using a Ubuntu 24.04 Deep Learning AMI, I just needed to
source /opt/pytorch/bin/activate 
pip install vllm==0.15.1
pip install huggingface-hub==0.36.0
pip install pydantic-settings==2.12.0
```

Once you have vLLM installed, you can install the package:

```bash
pip install -e .
```

For local development:

```bash
pip install -e .[dev]
```

## Configuration

Configuration is defined in `glassbox/config.py` and can be provided programmatically or through `glassbox.yaml`.

Example:

```yaml
scores_matrix:
  enabled: true
  interval: 32
  rank: 4
  method: randomized
  heads: [0]

degree_normalized_matrix:
  enabled: true
  interval: 32
  rank: 4
  method: randomized
  heads: [0]
  threshold: 2048
  block_size: 256
  hodge_target_cv: 0.05
  hodge_curl_seed: 42

output: experiments/results/svd_features.jsonl
```

Important knobs:

| Setting | Description |
|---|---|
| `scores_matrix.interval` | Snapshot cadence for `S = QK^T` |
| `scores_matrix.rank` | Number of singular values to keep |
| `scores_matrix.method` | `randomized` or `lanczos` |
| `scores_matrix.heads` | Heads to analyze |
| `degree_normalized_matrix.enabled` | Turn on normalized-operator analysis |
| `degree_normalized_matrix.threshold` | Sequence length cutoff for materialized vs matrix-free execution |
| `degree_normalized_matrix.block_size` | Row-block size for blocked operators |
| `output` | JSONL output path |

## Running the custom backend

### Test it on a single prompt

```bash
python -m glassbox.backends.runner \
  --model facebook/opt-125m \
  --interval 16 \
  --rank 4 \
  --heads 0 \
  --output svd_features.jsonl \
  --prompt "The future of artificial intelligence is"
```

This launches vLLM with:

- `attention_backend="CUSTOM"`
- `enforce_eager=True`

and writes inference-time snapshots to `svd_features.jsonl`.

### Run it in a vLLM server

```bash
vllm serve model --attention-backend CUSTOM
```

Registered via the `vllm.general_plugins` entry point -- vLLM loads it automatically.

### Run labeled extraction

```bash
python experiments/extract.py \
  --model Qwen/Qwen2-7B-Instruct \
  --request-type chat_completions \
  --dataset halueval \
  --max-samples 200 \
  --scores-matrix \
  --degree-normalized \
  --parquet
```

This produces:

- per-request sample metadata
- JSONL snapshot features
- optional wide Parquet features for downstream training or analysis

## Benchmarks

Comming soon.

