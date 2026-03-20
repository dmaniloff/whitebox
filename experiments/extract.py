"""Spectral feature extraction on labeled datasets.

Usage:
    python experiments/extract.py                          # default: HaluEval, OPT-125m
    python experiments/extract.py --mode evaluate          # Mode A: prefill evaluation
    python experiments/extract.py --model Qwen/Qwen2-7B-Instruct --request-type chat_completions
    python experiments/extract.py --max-samples 50         # quick test
    python experiments/extract.py --degree-normalized      # also compute M features
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path

import click

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "facebook/opt-125m"


def log(msg: str) -> None:
    click.echo(f"[spectral] {msg}")


# ── Dataset loading ───────────────────────────────────────────────────────


def load_halueval(max_samples: int) -> list[dict]:
    from datasets import load_dataset

    log("Loading HaluEval qa...")
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    indices = list(range(len(ds)))
    random.Random(42).shuffle(indices)
    samples = []
    for i in indices:
        if len(samples) >= max_samples:
            break
        row = ds[i]
        question = row["question"]
        # Each row produces 2 samples: right_answer (label=0) + hallucinated_answer (label=1)
        samples.append(
            {
                "idx": len(samples),
                "question": question,
                "response": row["right_answer"],
                "label": 0,
            }
        )
        if len(samples) >= max_samples:
            break
        samples.append(
            {
                "idx": len(samples),
                "question": question,
                "response": row["hallucinated_answer"],
                "label": 1,
            }
        )
    log(
        f"Loaded {len(samples)} samples ({sum(s['label'] for s in samples)} hallucinated)"
    )
    return samples


def load_truthfulqa(max_samples: int) -> list[dict]:
    from datasets import load_dataset

    log("Loading TruthfulQA...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    rng = random.Random(42)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    samples = []
    for i in indices:
        if len(samples) >= max_samples:
            break
        row = ds[i]
        question = row["question"]
        incorrect = row["incorrect_answers"]
        if not incorrect:
            continue
        # Pair best_answer (label=0) with a random incorrect_answer (label=1)
        samples.append(
            {
                "idx": len(samples),
                "question": question,
                "response": row["best_answer"],
                "label": 0,
            }
        )
        if len(samples) >= max_samples:
            break
        samples.append(
            {
                "idx": len(samples),
                "question": question,
                "response": rng.choice(incorrect),
                "label": 1,
            }
        )
    log(
        f"Loaded {len(samples)} samples ({sum(s['label'] for s in samples)} hallucinated)"
    )
    return samples


DATASET_LOADERS = {
    "halueval": load_halueval,
    "truthfulqa": load_truthfulqa,
}

SPECTRAL_FEATURES = ["sv_ratio", "sv1", "sv_entropy"]


def _write_parquet(svd_features_path: Path, samples_path: Path, out_path: Path) -> None:
    """Pivot JSONL results into shade-compatible wide Parquet.

    Output has one row per sample with columns:
        {feature}_L{layer}_H{head}  (e.g. sv_ratio_L0_H0)
        label, source, length
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Load SVD features
    svd_rows = []
    with open(svd_features_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            svs = row["singular_values"]
            row["sv_ratio"] = svs[0] / svs[1] if len(svs) >= 2 and svs[1] > 0 else None
            row["sv1"] = svs[0]
            total = sum(svs)
            if total > 0:
                ps = [s / total for s in svs]
                row["sv_entropy"] = -sum(p * math.log(p + 1e-12) for p in ps)
            else:
                row["sv_entropy"] = None
            if "hodge" in row and isinstance(row["hodge"], dict):
                for hk, hv in row["hodge"].items():
                    row[f"hodge_{hk}"] = hv
                del row["hodge"]
            del row["singular_values"]
            svd_rows.append(row)

    df_svd = pd.DataFrame(svd_rows)

    # Load sample metadata
    sample_rows = []
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if line:
                sample_rows.append(json.loads(line))
    df_samples = pd.DataFrame(sample_rows)

    # Join label + metadata onto SVD rows
    join_cols = ["request_id", "label"]
    for col in ["sample_id", "phase", "dataset", "prompt_length"]:
        if col in df_samples.columns:
            join_cols.append(col)
    df = df_svd.merge(df_samples[join_cols], on="request_id", how="left")

    # Identify sample key
    sample_col = "sample_id" if "sample_id" in df.columns else "request_id"

    # Determine features to pivot
    features = list(SPECTRAL_FEATURES)
    hodge_cols = [c for c in df.columns if c.startswith("hodge_") and df[c].notna().any()]
    features.extend(sorted(hodge_cols))

    # Aggregate: mean per (sample, signal, layer, head) across steps
    group_keys = [sample_col, "layer_idx", "head"]
    if "signal" in df.columns:
        group_keys.insert(1, "signal")
    agg = df.groupby(group_keys)[features].mean().reset_index()

    # Build signal prefix for column names (omit if only one signal)
    signals = agg["signal"].unique() if "signal" in agg.columns else [""]
    use_signal_prefix = len(signals) > 1

    # Pivot to wide format: one row per sample
    wide_rows = []
    for sample_id, sample_group in agg.groupby(sample_col):
        wide: dict = {}
        for _, r in sample_group.iterrows():
            li = int(r["layer_idx"])
            hi = int(r["head"])
            prefix = f"{r['signal']}_" if use_signal_prefix else ""
            for feat in features:
                val = r[feat]
                if pd.notna(val):
                    wide[f"{prefix}{feat}_L{li}_H{hi}"] = val
        wide[sample_col] = sample_id
        wide_rows.append(wide)

    df_wide = pd.DataFrame(wide_rows)

    # Join back sample-level metadata
    meta_cols = [sample_col, "label"]
    for col in ["dataset", "phase"]:
        if col in df_samples.columns:
            meta_cols.append(col)
    sample_meta = df.groupby(sample_col).first()[
        [c for c in ["label", "dataset", "L"] if c in df.columns]
    ].reset_index()
    if "dataset" in sample_meta.columns:
        sample_meta = sample_meta.rename(columns={"dataset": "source"})
    if "L" in sample_meta.columns:
        sample_meta = sample_meta.rename(columns={"L": "length"})
    df_wide = df_wide.merge(sample_meta, on=sample_col, how="left")

    pq.write_table(pa.Table.from_pandas(df_wide), out_path, compression="snappy")
    n_features = len([c for c in df_wide.columns if any(c.startswith(f) for f in features)])
    log(f"Parquet saved: {out_path} ({len(df_wide)} samples, {n_features} feature columns)")


# ── CLI ────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="HuggingFace model name."
)
@click.option(
    "--dataset",
    "dataset_name",
    default="halueval",
    show_default=True,
    type=click.Choice(list(DATASET_LOADERS.keys())),
    help="Dataset to use.",
)
@click.option(
    "--max-samples", default=200, show_default=True, help="Max samples to process."
)
@click.option(
    "--svd-interval", default=16, show_default=True, help="SVD snapshot interval."
)
@click.option("--svd-rank", default=4, show_default=True, help="SVD rank (k).")
@click.option(
    "--method",
    type=click.Choice(["randomized", "lanczos"]),
    default=None,
    help="SVD algorithm. [default: randomized]",
)
@click.option(
    "--heads",
    type=int,
    multiple=True,
    help="Head indices to analyze (repeatable). [default: [0]]",
)
@click.option(
    "--scores-matrix", "scores_matrix", is_flag=True, default=False,
    help="Compute scores-matrix SVD features.",
)
@click.option(
    "--degree-normalized", "degree_normalized", is_flag=True, default=False,
    help="Compute degree-normalized matrix features.",
)
@click.option(
    "--threshold",
    type=int,
    default=None,
    help="Seq length threshold for materialized vs matrix-free. [default: 2048]",
)
@click.option(
    "--hodge/--no-hodge",
    default=None,
    help="Compute Hodge decomposition features. [default: disabled]",
)
@click.option(
    "--request-type",
    "request_type",
    default="text_completions",
    show_default=True,
    type=click.Choice(["text_completions", "chat_completions"]),
    help="Request type (use chat_completions for instruct models).",
)
@click.option(
    "--max-tokens", default=128, show_default=True, help="Max tokens per completion."
)
@click.option(
    "--mode",
    default="generate",
    show_default=True,
    type=click.Choice(["generate", "evaluate"]),
    help="Mode: generate (Mode B) or evaluate (Mode A, prefill-only).",
)
@click.option(
    "--parquet", "parquet", is_flag=True, default=False,
    help="Also save results as wide Parquet (shade-compatible format).",
)
def main(
    model: str,
    dataset_name: str,
    max_samples: int,
    svd_interval: int,
    svd_rank: int,
    method: str | None,
    heads: tuple[int, ...],
    scores_matrix: bool,
    degree_normalized: bool,
    threshold: int | None,
    hodge: bool | None,
    request_type: str,
    max_tokens: int,
    mode: str,
    parquet: bool,
) -> None:
    """Run spectral feature extraction on a labeled dataset."""
    import vllm

    import glassbox.backends.svd_backend as svd_mod
    from glassbox.config import GlassboxConfig

    # Load dataset
    samples = DATASET_LOADERS[dataset_name](max_samples)

    if mode == "evaluate":
        # Mode A: prefill-only, SVD fires every token
        svd_interval = 1
        max_tokens = 1

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("experiments/results") / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    # Write config metadata
    config = {
        "model": model,
        "dataset": dataset_name,
        "max_samples": max_samples,
        "svd_interval": svd_interval,
        "svd_rank": svd_rank,
        "method": method or "randomized",
        "heads": list(heads) if heads else [0],
        "scores_matrix": scores_matrix,
        "degree_normalized": degree_normalized,
        "request_type": request_type,
        "max_tokens": max_tokens,
        "mode": mode,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    svd_features_path = outdir / "svd_features.jsonl"

    log(f"Results directory: {outdir}")
    log(f"Model: {model}")
    log(f"Mode: {mode}")
    log(f"Dataset: {dataset_name} ({len(samples)} samples)")
    log(f"SVD: interval={svd_interval}, rank={svd_rank}, method={method or 'randomized'}")
    if heads:
        log(f"Heads: {list(heads)}")
    if degree_normalized:
        log(f"Degree-normalized: enabled (threshold={threshold or 2048}, hodge={hodge or False})")

    if not scores_matrix and not degree_normalized:
        raise click.UsageError("At least one of --scores-matrix or --degree-normalized must be enabled.")

    # Configure glassbox backend
    gb_kwargs: dict = {"output": str(svd_features_path)}

    if scores_matrix:
        scores_cfg: dict = {"interval": svd_interval, "rank": svd_rank}
        if method is not None:
            scores_cfg["method"] = method
        if heads:
            scores_cfg["heads"] = list(heads)
        gb_kwargs["scores_matrix"] = scores_cfg

    if degree_normalized:
        dn_cfg: dict = {"enabled": True, "interval": svd_interval, "rank": svd_rank}
        if method is not None:
            dn_cfg["method"] = method
        if heads:
            dn_cfg["heads"] = list(heads)
        if threshold is not None:
            dn_cfg["threshold"] = threshold
        if hodge is not None:
            dn_cfg["hodge"] = hodge
        gb_kwargs["degree_normalized_matrix"] = dn_cfg

    gb_config = GlassboxConfig(**gb_kwargs)
    svd_mod.SVDTritonAttentionImpl.config = gb_config

    # Create vLLM engine
    log("Creating vLLM engine with CUSTOM attention backend")
    llm = vllm.LLM(
        model=model,
        attention_backend="CUSTOM",
        enforce_eager=True,
    )

    samples_path = outdir / "samples.jsonl"
    samples_f = open(samples_path, "w")

    request_counter = 0
    for i, sample in enumerate(samples):
        try:
            if mode == "evaluate":
                # Two-phase prefill: question-only baseline, then full
                prompt_q = f"Q: {sample['question']}\nA:"
                prompt_full = f"Q: {sample['question']}\nA: {sample['response']}"

                for phase, prompt in [
                    ("question", prompt_q),
                    ("full", prompt_full),
                ]:
                    outputs = llm.generate(
                        [prompt],
                        vllm.SamplingParams(max_tokens=1),
                    )
                    sample_row = {
                        "request_id": request_counter,
                        "sample_id": sample["idx"],
                        "phase": phase,
                        "dataset": dataset_name,
                        **sample,
                        "prompt_length": len(prompt),
                        "generated": outputs[0].outputs[0].text,
                    }
                    samples_f.write(json.dumps(sample_row) + "\n")
                    samples_f.flush()
                    request_counter += 1

            else:
                sampling_params = vllm.SamplingParams(max_tokens=max_tokens)
                if request_type == "chat_completions":
                    outputs = llm.chat(
                        messages=[{"role": "user", "content": sample["question"]}],
                        sampling_params=sampling_params,
                    )
                    generated = outputs[0].outputs[0].text
                    prompt = sample["question"]
                else:
                    prompt = f"Q: {sample['question']}\nA:"
                    outputs = llm.generate(
                        [prompt], sampling_params,
                    )
                    generated = outputs[0].outputs[0].text

                sample_row = {
                    "request_id": request_counter,
                    "sample_id": sample["idx"],
                    "phase": "generate",
                    "dataset": dataset_name,
                    **sample,
                    "prompt_length": len(prompt),
                    "generated": generated,
                    "response_length": len(sample.get("response", "")),
                }
                samples_f.write(json.dumps(sample_row) + "\n")
                samples_f.flush()
                request_counter += 1

        except Exception as e:
            log(f"  [{i + 1}/{len(samples)}] ERROR: {e}")
            continue

        label_str = "HALL" if sample["label"] == 1 else "OK"
        if (i + 1) % 10 == 0 or i == 0:
            log(f"  [{i + 1}/{len(samples)}] {label_str}")

    samples_f.close()
    log(f"Done! {len(samples)} samples, {request_counter} requests")
    log(f"  samples:      {samples_path}")
    log(f"  svd features: {svd_features_path}")

    if parquet:
        parquet_path = outdir / "features.parquet"
        _write_parquet(svd_features_path, samples_path, parquet_path)


if __name__ == "__main__":
    main()
