"""Spectral feature extraction and hallucination correlation analysis.

Usage:
    python experiments/spectral.py run                          # default: HaluEval, OPT-125m
    python experiments/spectral.py run --mode evaluate          # Mode A: prefill evaluation
    python experiments/spectral.py run --model Qwen/Qwen2-7B-Instruct --request-type chat_completions
    python experiments/spectral.py run --max-samples 50         # quick test

    python experiments/spectral.py analyze experiments/results/<timestamp>
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import click

# ── Constants ──────────────────────────────────────────────────────────────

VLLM = "/opt/pytorch/bin/vllm"
DEFAULT_MODEL = "facebook/opt-125m"
DEFAULT_PORT = 8000

SPECTRAL_FEATURES = ["sv_ratio", "sv1", "sv_entropy"]


# ── Shared helpers ─────────────────────────────────────────────────────────


def log(msg: str) -> None:
    click.echo(f"[spectral] {msg}")


def kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], stderr=subprocess.DEVNULL, text=True
        )
    except subprocess.CalledProcessError:
        return
    for pid in out.strip().split("\n"):
        if pid:
            os.kill(int(pid), signal.SIGKILL)
            log(f"Killed pid {pid} on port {port}")


def wait_for_server(port: int, timeout: int = 120) -> bool:
    log(f"Waiting for server on port {port}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            urlopen(f"http://localhost:{port}/health", timeout=2)
            elapsed = int(time.time() - start)
            log(f"Server ready after {elapsed}s")
            return True
        except Exception:
            time.sleep(2)
    log(f"ERROR: Server failed to start within {timeout}s")
    return False


# ── Dataset loading ───────────────────────────────────────────────────────


def load_halueval(max_samples: int) -> list[dict]:
    from datasets import load_dataset

    log("Loading HaluEval qa...")
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    samples = []
    for i, row in enumerate(ds):
        if len(samples) >= max_samples:
            break
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
    samples = []
    for row in ds:
        if len(samples) >= max_samples:
            break
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


# ── CLI ────────────────────────────────────────────────────────────────────


@click.group()
def cli():
    """Spectral feature extraction and hallucination correlation analysis."""


@cli.command()
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
@click.option("--port", default=DEFAULT_PORT, show_default=True, help="Server port.")
@click.option(
    "--svd-interval", default=16, show_default=True, help="SVD snapshot interval."
)
@click.option("--svd-rank", default=4, show_default=True, help="SVD rank (k).")
@click.option(
    "--request-type",
    "request_type",
    default="text_completions",
    show_default=True,
    type=click.Choice(["text_completions", "chat_completions"]),
    help="Request type (use chat_completions for instruct models).",
)
@click.option(
    "--server-timeout",
    default=300,
    show_default=True,
    help="Max seconds to wait for server.",
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
def run(
    model: str,
    dataset_name: str,
    max_samples: int,
    port: int,
    svd_interval: int,
    svd_rank: int,
    request_type: str,
    server_timeout: int,
    max_tokens: int,
    mode: str,
) -> None:
    """Run spectral feature extraction on a labeled dataset."""
    import openai

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
        "request_type": request_type,
        "max_tokens": max_tokens,
        "mode": mode,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    log(f"Results directory: {outdir}")
    log(f"Model: {model}")
    log(f"Mode: {mode}")
    log(f"Dataset: {dataset_name} ({len(samples)} samples)")
    log(f"SVD: interval={svd_interval}, rank={svd_rank}")

    # Set up environment — backend writes structured JSONL via GLASSBOX_SVD_OUTPUT
    svd_features_path = outdir / "svd_features.jsonl"
    env = os.environ.copy()
    env["GLASSBOX_SVD_INTERVAL"] = str(svd_interval)
    env["GLASSBOX_SVD_RANK"] = str(svd_rank)
    env["GLASSBOX_SVD_OUTPUT"] = str(svd_features_path)

    # Build and start server
    server_cmd = [
        VLLM,
        "serve",
        model,
        "--attention-backend",
        "CUSTOM",
        "--enforce-eager",
        "--port",
        str(port),
    ]

    kill_port(port)
    time.sleep(1)

    log(f"Starting server: {' '.join(server_cmd)}")
    server_log_path = outdir / "server.log"
    server_log_w = open(server_log_path, "w")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=server_log_w,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )

    try:
        if not wait_for_server(port, timeout=server_timeout):
            log("Server log tail:")
            server_log_w.flush()
            click.echo(server_log_path.read_text()[-2000:])
            os.killpg(server_proc.pid, signal.SIGKILL)
            sys.exit(1)

        client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")

        samples_path = outdir / "samples.jsonl"
        samples_f = open(samples_path, "w")

        request_counter = 0
        for i, sample in enumerate(samples):
            # Send request
            try:
                if mode == "evaluate":
                    # Mode A: send full (question + response) as prefill
                    prompt = f"Q: {sample['question']}\nA: {sample['response']}"
                    resp = client.completions.create(
                        model=model,
                        prompt=prompt,
                        max_tokens=1,
                    )
                    generated = resp.choices[0].text
                elif request_type == "chat_completions":
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": sample["question"]}],
                        max_tokens=max_tokens,
                    )
                    generated = resp.choices[0].message.content or ""
                else:
                    prompt = f"Q: {sample['question']}\nA:"
                    resp = client.completions.create(
                        model=model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                    )
                    generated = resp.choices[0].text
            except Exception as e:
                log(f"  [{i + 1}/{len(samples)}] ERROR: {e}")
                continue

            # Write sample metadata with request_id matching backend counter
            sample_row = {
                "request_id": request_counter,
                "dataset": dataset_name,
                **sample,
                "generated": generated,
                "response_length": len(sample.get("response", "")),
            }
            samples_f.write(json.dumps(sample_row) + "\n")
            samples_f.flush()

            request_counter += 1
            label_str = "HALL" if sample["label"] == 1 else "OK"
            if (i + 1) % 10 == 0 or i == 0:
                log(f"  [{i + 1}/{len(samples)}] {label_str}")

        samples_f.close()
        log(f"Done! {request_counter} samples processed")
        log(f"  samples:      {samples_path}")
        log(f"  svd features: {svd_features_path}")

    finally:
        os.killpg(server_proc.pid, signal.SIGKILL)
        server_proc.wait()
        server_log_w.close()
        kill_port(port)
        time.sleep(2)


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, help="Override plot output directory.")
def analyze(results_dir: str, output_dir: str | None) -> None:
    """Analyze spectral features and correlate with hallucination labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import pointbiserialr
    from sklearn.metrics import roc_auc_score

    base = Path(results_dir)
    plot_dir = Path(output_dir) if output_dir else base

    # ── Load data ─────────────────────────────────────────────────────────
    svd_path = base / "svd_features.jsonl"
    legacy_path = base / "features.jsonl"

    if svd_path.exists():
        log("Loading svd_features.jsonl (structured output from backend)")
        svd_rows = []
        with open(svd_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                svs = row["singular_values"]
                # Compute derived features from raw singular values
                row["sv_ratio"] = (
                    svs[0] / svs[1] if len(svs) >= 2 and svs[1] > 0 else None
                )
                row["sv1"] = svs[0]
                total = sum(svs)
                if total > 0:
                    ps = [s / total for s in svs]
                    row["sv_entropy"] = -sum(p * math.log(p + 1e-12) for p in ps)
                else:
                    row["sv_entropy"] = None
                del row["singular_values"]
                svd_rows.append(row)

        df_svd = pd.DataFrame(svd_rows)

        # Load sample metadata and join on request_id
        samples_path = base / "samples.jsonl"
        if not samples_path.exists():
            click.echo(f"No samples.jsonl found in {base}")
            sys.exit(1)

        sample_rows = []
        with open(samples_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_rows.append(json.loads(line))
        df_samples = pd.DataFrame(sample_rows)

        df = df_svd.merge(
            df_samples[["request_id", "label"]], on="request_id", how="left"
        )
        df = df.rename(columns={"request_id": "sample_idx"})

    elif legacy_path.exists():  # TODO: remove legacy path
        # Backward compat: load old log-parsed features.jsonl
        log("Loading features.jsonl (legacy log-parsed output)")
        rows = []
        with open(legacy_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row.pop("singular_values", None)
                    rows.append(row)
        df = pd.DataFrame(rows)
    else:
        click.echo(f"No svd_features.jsonl or features.jsonl found in {base}")
        sys.exit(1)

    # ── Basic stats ───────────────────────────────────────────────────────
    n_samples = df["sample_idx"].nunique()
    n_layers = df["layer_idx"].nunique()
    log(f"Loaded {len(df)} snapshots from {n_samples} samples across {n_layers} layers")

    labels = df.groupby("sample_idx")["label"].first()
    n_hall = int(labels.sum())
    n_ok = len(labels) - n_hall
    log(f"Label distribution: {n_hall} hallucinated, {n_ok} correct")

    if n_hall < 3 or n_ok < 3:
        click.echo("Too few samples in one class for meaningful analysis.")
        sys.exit(1)

    # Print config if available
    config_path = base / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
        log(
            f"Config: model={config.get('model')}, dataset={config.get('dataset')}, "
            f"interval={config.get('svd_interval')}, rank={config.get('svd_rank')}, "
            f"mode={config.get('mode', 'generate')}"
        )

    layer_ids = sorted(df["layer_idx"].unique())

    # ── Per-layer correlation table ───────────────────────────────────────
    # Aggregate: per (sample, layer), take mean of each feature.
    # Then correlate per-sample means with label.
    agg = (
        df.groupby(["sample_idx", "layer_idx"])[SPECTRAL_FEATURES].mean().reset_index()
    )
    agg = agg.merge(labels.reset_index(), on="sample_idx")

    click.echo("")
    click.echo("=" * 72)
    click.echo("Point-Biserial Correlations by Layer (sample-level means vs label)")
    click.echo("=" * 72)

    # Header
    feat_hdrs = "".join(f" | {f:>14s}" for f in SPECTRAL_FEATURES)
    click.echo(f"{'Layer':>6s}{feat_hdrs}")
    click.echo("-" * (8 + 17 * len(SPECTRAL_FEATURES)))

    corr_matrix = {}  # (layer_idx, feature) -> (r, p)
    for layer_idx in layer_ids:
        layer_agg = agg[agg["layer_idx"] == layer_idx]
        row_str = f"{layer_idx:>6d}"
        for feat in SPECTRAL_FEATURES:
            vals = layer_agg[feat].dropna()
            lbls = layer_agg.loc[vals.index, "label"]
            if len(vals) < 10 or lbls.nunique() < 2:
                row_str += f" | {'n/a':>14s}"
                continue
            r, p = pointbiserialr(lbls, vals)
            corr_matrix[(layer_idx, feat)] = (r, p)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            row_str += f" | {r:>+7.4f} {sig:<2s} {p:>4.3f}"
        click.echo(row_str)

    # ── Global (all-layer) correlation ────────────────────────────────────
    global_agg = df.groupby("sample_idx")[SPECTRAL_FEATURES].mean().reset_index()
    global_agg = global_agg.merge(labels.reset_index(), on="sample_idx")

    click.echo("")
    row_str = f"{'ALL':>6s}"
    for feat in SPECTRAL_FEATURES:
        vals = global_agg[feat].dropna()
        lbls = global_agg.loc[vals.index, "label"]
        if len(vals) < 10 or lbls.nunique() < 2:
            row_str += f" | {'n/a':>14s}"
            continue
        r, p = pointbiserialr(lbls, vals)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        row_str += f" | {r:>+7.4f} {sig:<2s} {p:>4.3f}"
    click.echo(row_str)
    click.echo("")

    # ── AUROC evaluation ─────────────────────────────────────────────────
    def bootstrap_auroc(y_true, y_score, n_bootstrap=1000, seed=42):
        """Compute AUROC with bootstrap 95% CI."""
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if len(np.unique(y_true)) < 2:
            return None, None, None
        auc = roc_auc_score(y_true, y_score)
        rng = np.random.RandomState(seed)
        aucs = []
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        if not aucs:
            return auc, None, None
        lo = np.percentile(aucs, 2.5)
        hi = np.percentile(aucs, 97.5)
        return auc, lo, hi

    click.echo("=" * 72)
    click.echo("AUROC by Layer (sample-level means)")
    click.echo("=" * 72)

    feat_hdrs = "".join(f" | {f:>20s}" for f in SPECTRAL_FEATURES)
    click.echo(f"{'Layer':>6s}{feat_hdrs}")
    click.echo("-" * (8 + 23 * len(SPECTRAL_FEATURES)))

    auroc_matrix = {}  # (layer_idx, feature) -> (auc, lo, hi)
    for layer_idx in layer_ids:
        layer_agg = agg[agg["layer_idx"] == layer_idx]
        row_str = f"{layer_idx:>6d}"
        for feat in SPECTRAL_FEATURES:
            vals = layer_agg[feat].dropna()
            lbls = layer_agg.loc[vals.index, "label"]
            if len(vals) < 10 or lbls.nunique() < 2:
                row_str += f" | {'n/a':>20s}"
                continue
            auc, lo, hi = bootstrap_auroc(lbls.values, vals.values)
            if auc is not None:
                auroc_matrix[(layer_idx, feat)] = (auc, lo, hi)
                if lo is not None:
                    row_str += f" | {auc:.3f} [{lo:.3f}-{hi:.3f}]"
                else:
                    row_str += f" | {auc:.3f} [n/a]"
            else:
                row_str += f" | {'n/a':>20s}"
        click.echo(row_str)

    # ── Aggregated AUROC (mean/max across layers per sample) ─────────────
    click.echo("")
    sample_layer_agg = agg.pivot_table(
        index="sample_idx",
        columns="layer_idx",
        values=SPECTRAL_FEATURES,
        aggfunc="mean",
    )
    sample_labels = labels.loc[sample_layer_agg.index].values

    for agg_name, agg_fn in [("mean", np.nanmean), ("max", np.nanmax)]:
        row_str = f"{agg_name:>6s}"
        for feat in SPECTRAL_FEATURES:
            feat_cols = [
                (feat, li) for li in layer_ids if (feat, li) in sample_layer_agg.columns
            ]
            if not feat_cols:
                row_str += f" | {'n/a':>20s}"
                continue
            vals_matrix = sample_layer_agg[feat_cols].values
            agg_vals = agg_fn(vals_matrix, axis=1)
            mask = ~np.isnan(agg_vals)
            if mask.sum() < 10 or len(np.unique(sample_labels[mask])) < 2:
                row_str += f" | {'n/a':>20s}"
                continue
            auc, lo, hi = bootstrap_auroc(sample_labels[mask], agg_vals[mask])
            if auc is not None and lo is not None:
                row_str += f" | {auc:.3f} [{lo:.3f}-{hi:.3f}]"
            elif auc is not None:
                row_str += f" | {auc:.3f} [n/a]"
            else:
                row_str += f" | {'n/a':>20s}"
        click.echo(row_str)
    click.echo("")

    # ── Plot 1: Correlation heatmap ───────────────────────────────────────
    sns.set_theme(style="whitegrid")

    corr_data = pd.DataFrame(index=layer_ids, columns=SPECTRAL_FEATURES, dtype=float)
    for (li, feat), (r, _) in corr_matrix.items():
        corr_data.loc[li, feat] = r

    fig, ax = plt.subplots(figsize=(6, max(4, len(layer_ids) * 0.5)))
    sns.heatmap(
        corr_data.astype(float),
        annot=True,
        fmt="+.3f",
        center=0,
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_ylabel("Layer")
    ax.set_xlabel("Feature")
    ax.set_title("Point-Biserial Correlation with Hallucination Label")
    plt.tight_layout()

    heatmap_path = str(plot_dir / "spectral_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Heatmap saved to {heatmap_path}")

    # ── Plot 2: AUROC heatmap ────────────────────────────────────────────
    if auroc_matrix:
        auroc_data = pd.DataFrame(
            index=layer_ids, columns=SPECTRAL_FEATURES, dtype=float
        )
        for (li, feat), (auc, _, _) in auroc_matrix.items():
            auroc_data.loc[li, feat] = auc

        fig, ax = plt.subplots(figsize=(6, max(4, len(layer_ids) * 0.5)))
        sns.heatmap(
            auroc_data.astype(float),
            annot=True,
            fmt=".3f",
            center=0.5,
            cmap="RdYlGn",
            vmin=0.3,
            vmax=0.7,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_ylabel("Layer")
        ax.set_xlabel("Feature")
        ax.set_title("AUROC by Layer and Feature")
        plt.tight_layout()

        auroc_heatmap_path = str(plot_dir / "spectral_auroc_heatmap.png")
        plt.savefig(auroc_heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"AUROC heatmap saved to {auroc_heatmap_path}")

    # ── Plot 3: Per-layer distributions ───────────────────────────────────
    # Melt to long form for seaborn
    df["label_str"] = df["label"].map({0: "Correct", 1: "Hallucinated"})
    plot_df = df.melt(
        id_vars=["sample_idx", "layer_idx", "label_str"],
        value_vars=SPECTRAL_FEATURES,
        var_name="feature",
        value_name="value",
    ).dropna(subset=["value"])

    n_features = len(SPECTRAL_FEATURES)
    fig, axes = plt.subplots(
        n_features,
        1,
        figsize=(max(10, len(layer_ids) * 0.8), 4 * n_features),
        sharex=True,
    )
    if n_features == 1:
        axes = [axes]

    for ax, feat in zip(axes, SPECTRAL_FEATURES):
        feat_df = plot_df[plot_df["feature"] == feat]
        sns.stripplot(
            data=feat_df,
            x="layer_idx",
            y="value",
            hue="label_str",
            dodge=True,
            alpha=0.25,
            size=2,
            jitter=0.3,
            ax=ax,
            legend=False,
        )
        sns.violinplot(
            data=feat_df,
            x="layer_idx",
            y="value",
            hue="label_str",
            split=True,
            inner="quart",
            cut=0,
            density_norm="width",
            alpha=0.6,
            ax=ax,
        )
        ax.set_ylabel(feat)
        ax.set_xlabel("")
        ax.legend(title="", loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Layer")
    fig.suptitle(
        "Spectral Feature Distributions by Layer and Label",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    dist_path = str(plot_dir / "spectral_distributions.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Distributions saved to {dist_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    if corr_matrix:
        best_key = max(corr_matrix, key=lambda k: abs(corr_matrix[k][0]))
        r_val, p_val = corr_matrix[best_key]
        log(
            f"Strongest per-layer correlation: layer {best_key[0]} / {best_key[1]} "
            f"(r={r_val:+.4f}, p={p_val:.4f})"
        )
    if auroc_matrix:
        best_auc_key = max(auroc_matrix, key=lambda k: abs(auroc_matrix[k][0] - 0.5))
        auc_val, auc_lo, auc_hi = auroc_matrix[best_auc_key]
        ci_str = f" [{auc_lo:.3f}-{auc_hi:.3f}]" if auc_lo is not None else ""
        log(
            f"Best per-layer AUROC: layer {best_auc_key[0]} / {best_auc_key[1]} "
            f"(AUROC={auc_val:.3f}{ci_str})"
        )
    click.echo("")


if __name__ == "__main__":
    cli()
