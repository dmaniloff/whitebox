"""Hallucination correlation analysis for spectral features.

Usage:
    python experiments/analyze.py experiments/results/<timestamp>
    python experiments/analyze.py experiments/results/<timestamp> --output-dir plots/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from glassbox.results import SPECTRAL_FEATURE_NAMES, SVDSnapshot

# ── Constants ──────────────────────────────────────────────────────────────

LABEL_COLORS = {0: "#1565C0", 1: "#C62828"}
LABEL_NAMES = {0: "Correct", 1: "Hallucinated"}


def log(msg: str) -> None:
    click.echo(f"[spectral] {msg}")


# ── Plot helpers ──────────────────────────────────────────────────────────


def plot_violin_pointrange(
    df,
    features: list[str],
    feat_labels: list[str],
    layer_ids: list[int],
    title: str,
    out_path: str,
    show_zero_line: bool = False,
) -> None:
    """Half-violin + pointrange + strip plot, split by label.

    Expects *df* to have columns: sample_idx, layer_idx, label, L, and each
    feature in *features*.  Dot size encodes sequence length (L).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_feats = len(features)
    fig, axes = plt.subplots(
        n_feats, 1,
        figsize=(max(12, len(layer_ids) * 1.0), 4.5 * n_feats),
    )
    if n_feats == 1:
        axes = [axes]

    # Sequence-length → dot size mapping
    L_vals = df["L"].dropna()
    L_min, L_max = (L_vals.min(), L_vals.max()) if len(L_vals) else (1, 1)
    s_min, s_max = 15, 120

    def _size(L):
        t = (L - L_min) / max(L_max - L_min, 1)
        return s_min + t * (s_max - s_min)

    for ax_i, (feat, feat_label) in enumerate(zip(features, feat_labels)):
        ax = axes[ax_i]

        for label_val, side in [(0, "left"), (1, "right")]:
            color = LABEL_COLORS[label_val]
            sub = df[df["label"] == label_val]

            # Collect per-layer arrays
            layer_data, layer_pos, layer_pts = [], [], []
            for li_i, li in enumerate(layer_ids):
                chunk = sub[sub["layer_idx"] == li]
                vals = chunk[feat].dropna().values
                if len(vals) == 0:
                    continue
                layer_data.append(vals)
                layer_pos.append(li_i)
                Ls = chunk.loc[chunk[feat].notna(), "L"].values
                layer_pts.append(list(zip(vals, Ls)))

            if not layer_data:
                continue

            # ── Half violin ───────────────────────────────────────────
            parts = ax.violinplot(
                layer_data, positions=layer_pos,
                widths=0.8, showextrema=False, showmedians=False,
            )
            for pc in parts["bodies"]:
                verts = pc.get_paths()[0].vertices
                m = np.mean(verts[:, 0])
                if side == "left":
                    verts[:, 0] = np.clip(verts[:, 0], -np.inf, m)
                else:
                    verts[:, 0] = np.clip(verts[:, 0], m, np.inf)
                pc.set_facecolor(color)
                pc.set_alpha(0.2)
                pc.set_edgecolor(color)
                pc.set_linewidth(0.8)

            # ── Pointrange (thin whisker, thick IQR, median dot) ──────
            offset = -0.13 if side == "left" else 0.13
            for li_i, vals in zip(layer_pos, layer_data):
                if len(vals) < 2:
                    ax.plot(li_i + offset, vals[0], "o", color=color, ms=5)
                    continue
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                iqr = q3 - q1
                lo = max(vals.min(), q1 - 1.5 * iqr)
                hi = min(vals.max(), q3 + 1.5 * iqr)
                x = li_i + offset
                ax.plot([x, x], [lo, hi], color=color, lw=1,
                        solid_capstyle="round", zorder=5)
                ax.plot([x, x], [q1, q3], color=color, lw=4.5,
                        solid_capstyle="round", alpha=0.7, zorder=6)
                ax.plot(x, med, "o", color="white", ms=4.5,
                        mec=color, mew=1.2, zorder=7)

            # ── Strip dots (size = seq length) ────────────────────────
            jitter_base = -0.28 if side == "left" else 0.22
            rng = np.random.RandomState(42 + label_val)
            for li_i, pts in zip(layer_pos, layer_pts):
                for val, L in pts:
                    jx = jitter_base + rng.uniform(-0.08, 0.08)
                    ax.scatter(
                        li_i + jx, val, c=color, s=_size(L), alpha=0.5,
                        edgecolors="white", linewidths=0.3, zorder=4,
                    )

            # Legend entry (first panel only)
            if ax_i == 0:
                ax.plot([], [], color=color, lw=6, alpha=0.5,
                        label=LABEL_NAMES[label_val])

        if show_zero_line:
            ax.axhline(y=0, color="gray", lw=0.8, ls="--", alpha=0.5)

        ax.set_ylabel(feat_label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.tick_params(labelsize=10)
        ax.set_xticks(range(len(layer_ids)))
        ax.set_xticklabels(
            [f"L{li}" for li in layer_ids] if ax_i == n_feats - 1 else [],
            fontsize=10,
        )

        if ax_i == 0:
            # Size legend
            for L_ex, lbl in [
                (int(L_min), f"L={int(L_min)}"),
                (int((L_min + L_max) // 2), f"L={int((L_min + L_max) // 2)}"),
                (int(L_max), f"L={int(L_max)}"),
            ]:
                ax.scatter([], [], c="gray", s=_size(L_ex), alpha=0.6,
                           edgecolors="white", linewidths=0.3, label=lbl)
            ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
                      ncol=2, columnspacing=0.8, handletextpad=0.3)

    axes[-1].set_xlabel("Layer", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Distributions saved to {out_path}")


# ── Analysis helpers ──────────────────────────────────────────────────────


def bootstrap_auroc(y_true, y_score, n_bootstrap=1000, seed=42):
    """Compute AUROC with bootstrap 95% CI."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

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
    return auc, np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def _analyze_signal(
    df,
    df_all,
    features: list[str],
    feat_plot_labels: list[str],
    signal_name: str,
    layer_ids: list[int],
    labels,
    plot_dir: Path,
    has_phases: bool,
) -> None:
    """Run correlation, AUROC, and distribution analysis for one signal type."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import pointbiserialr

    signal_label = signal_name.replace("_", " ").title()

    n_samples = df["sample_idx"].nunique()
    n_layers = df["layer_idx"].nunique()
    log(f"[{signal_label}] {len(df)} snapshots from {n_samples} samples across {n_layers} layers")

    # ── Per-layer correlation table ───────────────────────────────────────
    agg = df.groupby(["sample_idx", "layer_idx"])[features].mean().reset_index()
    agg = agg.merge(labels.reset_index(), on="sample_idx")

    click.echo("")
    click.echo("=" * 72)
    click.echo(f"Point-Biserial Correlations — {signal_label}")
    click.echo("=" * 72)

    feat_hdrs = "".join(f" | {f:>14s}" for f in features)
    click.echo(f"{'Layer':>6s}{feat_hdrs}")
    click.echo("-" * (8 + 17 * len(features)))

    corr_matrix = {}
    for layer_idx in layer_ids:
        layer_agg = agg[agg["layer_idx"] == layer_idx]
        row_str = f"{layer_idx:>6d}"
        for feat in features:
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
    global_agg = df.groupby("sample_idx")[features].mean().reset_index()
    global_agg = global_agg.merge(labels.reset_index(), on="sample_idx")

    click.echo("")
    row_str = f"{'ALL':>6s}"
    for feat in features:
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

    # ── AUROC by layer ────────────────────────────────────────────────────
    click.echo("=" * 72)
    click.echo(f"AUROC by Layer — {signal_label}")
    click.echo("=" * 72)

    feat_hdrs = "".join(f" | {f:>20s}" for f in features)
    click.echo(f"{'Layer':>6s}{feat_hdrs}")
    click.echo("-" * (8 + 23 * len(features)))

    auroc_matrix = {}
    for layer_idx in layer_ids:
        layer_agg = agg[agg["layer_idx"] == layer_idx]
        row_str = f"{layer_idx:>6d}"
        for feat in features:
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
        index="sample_idx", columns="layer_idx", values=features, aggfunc="mean",
    )
    sample_labels = labels.loc[sample_layer_agg.index].values

    for agg_name, agg_fn in [("mean", np.nanmean), ("max", np.nanmax)]:
        row_str = f"{agg_name:>6s}"
        for feat in features:
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

    # ── Plots ─────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")

    # Correlation heatmap
    corr_data = pd.DataFrame(index=layer_ids, columns=features, dtype=float)
    for (li, feat), (r, _) in corr_matrix.items():
        corr_data.loc[li, feat] = r

    fig, ax = plt.subplots(
        figsize=(max(6, len(features) * 1.5), max(4, len(layer_ids) * 0.5))
    )
    sns.heatmap(
        corr_data.astype(float), annot=True, fmt="+.3f", center=0,
        cmap="RdBu_r", vmin=-0.5, vmax=0.5, ax=ax, linewidths=0.5,
    )
    ax.set_ylabel("Layer")
    ax.set_xlabel("Feature")
    ax.set_title(f"Point-Biserial Correlation — {signal_label}")
    plt.tight_layout()
    plt.savefig(str(plot_dir / f"{signal_name}_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Heatmap saved to {signal_name}_heatmap.png")

    # AUROC heatmap
    if auroc_matrix:
        auroc_data = pd.DataFrame(index=layer_ids, columns=features, dtype=float)
        for (li, feat), (auc, _, _) in auroc_matrix.items():
            auroc_data.loc[li, feat] = auc

        fig, ax = plt.subplots(
            figsize=(max(6, len(features) * 1.5), max(4, len(layer_ids) * 0.5))
        )
        sns.heatmap(
            auroc_data.astype(float), annot=True, fmt=".3f", center=0.5,
            cmap="RdYlGn", vmin=0.3, vmax=0.7, ax=ax, linewidths=0.5,
        )
        ax.set_ylabel("Layer")
        ax.set_xlabel("Feature")
        ax.set_title(f"AUROC by Layer — {signal_label}")
        plt.tight_layout()
        plt.savefig(
            str(plot_dir / f"{signal_name}_auroc_heatmap.png"), dpi=150, bbox_inches="tight",
        )
        plt.close()
        log(f"AUROC heatmap saved to {signal_name}_auroc_heatmap.png")

    # Distribution plot
    df_plot = df.copy()
    if "L" not in df_plot.columns:
        df_plot["L"] = 0
    dist_df = df_plot[["sample_idx", "layer_idx", "label", "L"] + features].copy()
    plot_violin_pointrange(
        dist_df, features=features, feat_labels=feat_plot_labels,
        layer_ids=layer_ids,
        title=f"Feature Distributions — {signal_label}",
        out_path=str(plot_dir / f"{signal_name}_distributions.png"),
    )

    # Feature vs length scatter
    if "L" in df.columns and corr_matrix:
        best_layer = max(corr_matrix, key=lambda k: abs(corr_matrix[k][0]))[0]
        layer_df = df[df["layer_idx"] == best_layer].copy()
        layer_df["label_str"] = layer_df["label"].map({0: "Correct", 1: "Hallucinated"})
        n_feats = len(features)
        fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 4))
        if n_feats == 1:
            axes = [axes]
        for ax, feat in zip(axes, features):
            sns.scatterplot(
                data=layer_df, x="L", y=feat, hue="label_str",
                alpha=0.4, s=15, ax=ax,
            )
            ax.set_title(f"Layer {best_layer}")
            ax.set_xlabel("Sequence Length (tokens)")
        plt.suptitle(
            f"Features vs Sequence Length — {signal_label}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            str(plot_dir / f"{signal_name}_feature_vs_length_layer{best_layer}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        log(f"Feature vs length scatter saved to {signal_name}_feature_vs_length_layer{best_layer}.png")

    # ── Two-phase analysis (evaluate mode) ────────────────────────────────
    if has_phases and df_all is not None:
        click.echo("=" * 72)
        click.echo(f"Two-Phase Analysis: Question vs Full — {signal_label}")
        click.echo("=" * 72)

        df_q = df_all[df_all["phase"] == "question"].copy()
        df_f = df  # already filtered to "full"

        agg_q = (
            df_q.groupby(["sample_idx", "layer_idx"])[features].mean().reset_index()
        )
        agg_f = (
            df_f.groupby(["sample_idx", "layer_idx"])[features].mean().reset_index()
        )

        merged_phases = agg_q.merge(
            agg_f, on=["sample_idx", "layer_idx"], suffixes=("_q", "_f")
        )
        delta_features = [f"{f}_delta" for f in features]
        for feat in features:
            merged_phases[f"{feat}_delta"] = (
                merged_phases[f"{feat}_f"] - merged_phases[f"{feat}_q"]
            )

        merged_phases = merged_phases.merge(labels.reset_index(), on="sample_idx")

        # Delta correlation table
        click.echo("")
        click.echo("Delta Correlations (full - question) vs label:")
        feat_hdrs = "".join(f" | {f:>20s}" for f in delta_features)
        click.echo(f"{'Layer':>6s}{feat_hdrs}")
        click.echo("-" * (8 + 23 * len(delta_features)))

        delta_corr = {}
        for layer_idx in layer_ids:
            lm = merged_phases[merged_phases["layer_idx"] == layer_idx]
            row_str = f"{layer_idx:>6d}"
            for feat in delta_features:
                vals = lm[feat].dropna()
                lbls = lm.loc[vals.index, "label"]
                if len(vals) < 10 or lbls.nunique() < 2:
                    row_str += f" | {'n/a':>20s}"
                    continue
                r, p = pointbiserialr(lbls, vals)
                delta_corr[(layer_idx, feat)] = (r, p)
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                row_str += f" | {r:>+7.4f} {sig:<2s}  {p:>5.3f}"
            click.echo(row_str)

        # Global delta row
        click.echo("")
        global_delta = (
            merged_phases.groupby("sample_idx")[delta_features].mean().reset_index()
        )
        global_delta = global_delta.merge(labels.reset_index(), on="sample_idx")
        row_str = f"{'ALL':>6s}"
        for feat in delta_features:
            vals = global_delta[feat].dropna()
            lbls = global_delta.loc[vals.index, "label"]
            if len(vals) < 10 or lbls.nunique() < 2:
                row_str += f" | {'n/a':>20s}"
                continue
            r, p = pointbiserialr(lbls, vals)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            row_str += f" | {r:>+7.4f} {sig:<2s}  {p:>5.3f}"
        click.echo(row_str)

        # Delta AUROC
        click.echo("")
        click.echo("Delta AUROC (full - question) by Layer:")
        feat_hdrs = "".join(f" | {f:>20s}" for f in delta_features)
        click.echo(f"{'Layer':>6s}{feat_hdrs}")
        click.echo("-" * (8 + 23 * len(delta_features)))

        for layer_idx in layer_ids:
            lm = merged_phases[merged_phases["layer_idx"] == layer_idx]
            row_str = f"{layer_idx:>6d}"
            for feat in delta_features:
                vals = lm[feat].dropna()
                lbls = lm.loc[vals.index, "label"]
                if len(vals) < 10 or lbls.nunique() < 2:
                    row_str += f" | {'n/a':>20s}"
                    continue
                auc, lo, hi = bootstrap_auroc(lbls.values, vals.values)
                if auc is not None and lo is not None:
                    row_str += f" | {auc:.3f} [{lo:.3f}-{hi:.3f}]"
                elif auc is not None:
                    row_str += f" | {auc:.3f} [n/a]"
                else:
                    row_str += f" | {'n/a':>20s}"
            click.echo(row_str)
        click.echo("")

        # Plot: Delta correlation heatmap
        if delta_corr:
            delta_corr_data = pd.DataFrame(
                index=layer_ids, columns=delta_features, dtype=float
            )
            for (li, feat), (r, _) in delta_corr.items():
                delta_corr_data.loc[li, feat] = r

            fig, ax = plt.subplots(
                figsize=(max(8, len(delta_features) * 2), max(4, len(layer_ids) * 0.5))
            )
            sns.heatmap(
                delta_corr_data.astype(float), annot=True, fmt="+.3f", center=0,
                cmap="RdBu_r", vmin=-0.5, vmax=0.5, ax=ax, linewidths=0.5,
            )
            ax.set_ylabel("Layer")
            ax.set_xlabel("Feature (full - question)")
            ax.set_title(f"Delta Correlation — {signal_label}")
            plt.tight_layout()
            plt.savefig(
                str(plot_dir / f"{signal_name}_delta_heatmap.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close()
            log(f"Delta heatmap saved to {signal_name}_delta_heatmap.png")

        # Plot: Delta distributions
        L_by_sample = (
            df_f.groupby("sample_idx")["L"].max().reset_index()
            if "L" in df_f.columns
            else pd.DataFrame(
                {"sample_idx": merged_phases["sample_idx"].unique(), "L": 0}
            )
        )
        delta_dist_df = merged_phases[
            ["sample_idx", "layer_idx", "label"] + delta_features
        ].merge(L_by_sample, on="sample_idx", how="left")

        delta_feat_labels = [f"\u0394 {fl}" for fl in feat_plot_labels]
        plot_violin_pointrange(
            delta_dist_df,
            features=delta_features,
            feat_labels=delta_feat_labels,
            layer_ids=layer_ids,
            title=f"Delta (Full \u2212 Question) Distributions \u2014 {signal_label}",
            out_path=str(plot_dir / f"{signal_name}_delta_distributions.png"),
            show_zero_line=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    if corr_matrix:
        best_key = max(corr_matrix, key=lambda k: abs(corr_matrix[k][0]))
        r_val, p_val = corr_matrix[best_key]
        log(
            f"[{signal_label}] Strongest correlation: layer {best_key[0]} / {best_key[1]} "
            f"(r={r_val:+.4f}, p={p_val:.4f})"
        )
    if auroc_matrix:
        best_auc_key = max(auroc_matrix, key=lambda k: abs(auroc_matrix[k][0] - 0.5))
        auc_val, auc_lo, auc_hi = auroc_matrix[best_auc_key]
        ci_str = f" [{auc_lo:.3f}-{auc_hi:.3f}]" if auc_lo is not None else ""
        log(
            f"[{signal_label}] Best AUROC: layer {best_auc_key[0]} / {best_auc_key[1]} "
            f"(AUROC={auc_val:.3f}{ci_str})"
        )


# ── CLI ────────────────────────────────────────────────────────────────────


@click.command()
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", default=None, help="Override plot output directory.")
def main(results_dir: str, output_dir: str | None) -> None:
    """Analyze spectral features and correlate with hallucination labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    base = Path(results_dir)
    plot_dir = Path(output_dir) if output_dir else base

    # ── Load data ─────────────────────────────────────────────────────────
    svd_path = base / "svd_features.jsonl"
    if not svd_path.exists():
        click.echo(f"No svd_features.jsonl found in {base}")
        sys.exit(1)

    log("Loading svd_features.jsonl")
    svd_rows = []
    with open(svd_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            snap = SVDSnapshot.from_jsonl_row(json.loads(line))
            row = {
                "signal": snap.feature_group,
                "request_id": snap.request_id,
                "layer": snap.layer,
                "layer_idx": snap.layer_idx,
                "head": snap.head,
                "step": snap.step,
                "L": snap.L,
            }
            if snap.tier is not None:
                row["tier"] = snap.tier
            # Flatten features to top-level columns
            feat_dict = snap.features.model_dump(exclude_none=True)
            for k, v in feat_dict.items():
                if k in SPECTRAL_FEATURE_NAMES:
                    row[k] = v
                else:
                    row[f"hodge_{k}"] = v
            svd_rows.append(row)

    df_all = pd.DataFrame(svd_rows)

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

    # Join: always label, plus phase/sample_id/prompt_length if present
    join_cols = ["request_id", "label"]
    for col in ["sample_id", "phase", "prompt_length"]:
        if col in df_samples.columns:
            join_cols.append(col)

    df_all = df_all.merge(df_samples[join_cols], on="request_id", how="left")

    has_phases = "phase" in df_all.columns and {"question", "full"} <= set(
        df_all["phase"].unique()
    )

    # Normalize sample_idx column
    if "sample_id" in df_all.columns:
        df_all = df_all.rename(columns={"sample_id": "sample_idx"})
    elif "sample_idx" not in df_all.columns:
        df_all = df_all.rename(columns={"request_id": "sample_idx"})

    # ── Basic stats ───────────────────────────────────────────────────────
    # Use "full" phase rows (or all rows if no phases) for stats
    if has_phases:
        df_stats = df_all[df_all["phase"] == "full"]
    else:
        df_stats = df_all

    n_samples = df_stats["sample_idx"].nunique()
    log(f"Loaded {len(df_stats)} total snapshots from {n_samples} samples")

    labels = df_stats.groupby("sample_idx")["label"].first()
    n_hall = int(labels.sum())
    n_ok = len(labels) - n_hall
    log(f"Label distribution: {n_hall} hallucinated, {n_ok} correct")

    if has_phases:
        log("Two-phase data detected (question + full); main tables use 'full' phase")

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
        if config.get("degree_normalized"):
            log("  degree_normalized=True")

    layer_ids = sorted(df_stats["layer_idx"].unique())

    # ── Per-signal analysis ───────────────────────────────────────────────
    signals = sorted(df_all["signal"].unique()) if "signal" in df_all.columns else ["scores_matrix"]
    log(f"Signals found: {signals}")

    for signal_name in signals:
        # Filter to this signal
        if "signal" in df_all.columns:
            df_sig_all = df_all[df_all["signal"] == signal_name].copy()
        else:
            df_sig_all = df_all.copy()

        if has_phases:
            df_sig = df_sig_all[df_sig_all["phase"] == "full"].copy()
        else:
            df_sig = df_sig_all.copy()

        if df_sig.empty:
            log(f"[{signal_name}] No data, skipping")
            continue

        # Determine features for this signal
        features = list(SPECTRAL_FEATURE_NAMES)
        feat_labels = ["\u03c3\u2081/\u03c3\u2082 Ratio", "\u03c3\u2081 (Leading SV)", "SV Entropy"]

        # Add hodge features if present (degree_normalized_matrix with hodge=True)
        hodge_cols = sorted([
            c for c in df_sig.columns
            if c.startswith("hodge_") and df_sig[c].notna().any()
        ])
        if hodge_cols:
            features.extend(hodge_cols)
            feat_labels.extend([
                c.replace("hodge_", "").replace("_", " ").title()
                for c in hodge_cols
            ])

        _analyze_signal(
            df_sig, df_sig_all, features, feat_labels,
            signal_name, layer_ids, labels, plot_dir, has_phases,
        )

    # ── Prompt length analysis (signal-independent) ───────────────────────
    if has_phases:
        df_len = df_all[df_all["phase"] == "full"]
    else:
        df_len = df_all
    # Use any signal's L column (same across signals for a given sample)
    if "L" in df_len.columns:
        sample_length = df_len.groupby("sample_idx")["L"].max().reset_index()
        sample_length = sample_length.merge(labels.reset_index(), on="sample_idx")
        sample_length["label_str"] = sample_length["label"].map(
            {0: "Correct", 1: "Hallucinated"}
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        for lv, ls, c in [(0, "Correct", "steelblue"), (1, "Hallucinated", "tomato")]:
            subset = sample_length[sample_length["label"] == lv]
            ax.hist(subset["L"], bins=30, alpha=0.5, label=ls, color=c)
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_ylabel("Count")
        ax.set_title("Sequence Length Distribution by Label")
        ax.legend()
        plt.tight_layout()
        len_dist_path = str(plot_dir / "seq_length_dist.png")
        plt.savefig(len_dist_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Sequence length distribution saved to {len_dist_path}")

    click.echo("")


if __name__ == "__main__":
    main()
