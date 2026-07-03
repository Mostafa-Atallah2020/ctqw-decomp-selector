"""Comparative data analysis of the ER and structured corpora.

All feature-based analysis is standardized (z-scored) so no feature dominates
by numeric scale: importance and PCA use standardized inputs, and the
distribution/scatter plots show pooled z-scores. Correlation is scale-invariant
already. Produces both a printed/JSON balance report and a set of comparative
plots (feature importance, label distribution, feature distributions,
correlation heatmaps, feature-vs-target scatter, and a PCA projection).

Reads data/<corpus>/features/graph_features.csv joined to
data/<corpus>/labels/labels.csv on graph_id.

This module is the importable library; run it via scripts/analyze_corpora.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = _REPO_ROOT / "data"
DEFAULT_PLOTS = _REPO_ROOT / "data" / "analysis"

# IEEE column widths, in inches (2-column template).
IEEE_COL_WIDTH = 3.5    # single column
IEEE_PAGE_WIDTH = 7.16  # full page (two columns)


def set_ieee_style() -> None:
    """Configure matplotlib for IEEE-paper figures: serif fonts, ~8pt, vector
    PDF with embedded fonts. Called once before plotting."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # embed TrueType fonts so the PDF renders identically everywhere.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    })

# Features to analyze (the model's inputs). Booleans/degenerate ones included so
# the report can flag them.
FEATURE_COLS = [
    "n_vertices", "edge_density", "avg_degree", "max_degree",
    "min_degree", "degree_variance", "max_matching_size", "spectral_gap",
    "avg_clustering", "triangle_count", "diameter",
    "is_bipartite", "is_tree", "is_regular",
]

log = logging.getLogger("analysis")

# Corpora shown in the analysis, in order, with their plot colors. The hybrid
# is the balanced subset drawn from er + structured.
CORPORA = [
    ("er", "tab:blue"),
    ("structured", "tab:orange"),
    ("hybrid", "tab:green"),
]
CORPUS_NAMES = [c for c, _ in CORPORA]
CORPUS_COLOR = dict(CORPORA)
# Display names for plot titles/legends.
CORPUS_LABEL = {
    "er": "Erdős-Rényi",
    "structured": "structured",
    "hybrid": "hybrid",
}
# Distinct marker per corpus, so scatter/PCA points are distinguishable by shape
# (not just color) even where they overlap or in grayscale.
CORPUS_MARKER = {
    "er": "o",        # circle
    "structured": "^",  # triangle
    "hybrid": "x",    # cross
}


def load_corpus(data_dir: Path, corpus: str):
    """Load features joined to labels for one corpus. Returns a DataFrame."""
    import pandas as pd
    feat = pd.read_csv(data_dir / corpus / "features" / "graph_features.csv")
    lab = pd.read_csv(data_dir / corpus / "labels" / "labels.csv")
    df = feat.merge(lab[["graph_id", "delta_cx", "delta_depth",
                         "cx_matching", "cx_pauli"]],
                    on="graph_id", how="inner")
    df["corpus"] = corpus
    df["winner"] = df["delta_cx"].apply(
        lambda d: "matching" if d > 0 else ("pauli" if d < 0 else "tie"))
    return df


def standardize(df):
    """Add z-scored `<feature>_z` columns for the non-degenerate features.

    All feature-based analysis (importance, PCA, distributions, scatter) uses
    these standardized columns so no feature dominates by numeric scale and the
    axes are comparable across features. Standardization is per corpus. (For
    correlation this is a no-op since Pearson r is scale-invariant, but we keep
    it uniform.)
    """
    import numpy as np
    cols = _model_features()
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        sd = v.std()
        df[f"{c}_z"] = (v - v.mean()) / (sd if sd > 0 else 1.0)
    return df


# ---------------------------------------------------------------------------
# balance report
# ---------------------------------------------------------------------------

def balance_report(df, corpus: str) -> dict:
    """Quantify label balance and feature-coverage balance for one corpus."""
    n = len(df)
    matching = int((df.delta_cx > 0).sum())
    pauli = int((df.delta_cx < 0).sum())
    ties = int((df.delta_cx == 0).sum())

    # Label balance: how far from 50/50. imbalance_ratio = majority/minority.
    maj, mino = max(matching, pauli), max(min(matching, pauli), 1)
    imbalance_ratio = round(maj / mino, 1)
    minority_frac = round(min(matching, pauli) / n, 4) if n else 0.0

    # Per-n win rate (the crossover story).
    by_n = {}
    for nv, s in df.groupby("n_vertices")["delta_cx"]:
        by_n[int(nv)] = {
            "count": int(len(s)),
            "matching_win_rate": round(float((s > 0).mean()), 3),
            "mean_delta_cx": round(float(s.mean()), 1),
        }

    # Feature coverage: fraction of graphs at each feature's modal value.
    degenerate = {}
    for f in FEATURE_COLS:
        if f not in df.columns:
            continue
        vc = df[f].value_counts(normalize=True)
        mode_frac = round(float(vc.iloc[0]), 3) if len(vc) else 1.0
        if mode_frac >= 0.90:  # flag near-constant features
            degenerate[f] = mode_frac

    # A blunt verdict, so the report is actionable at a glance.
    if minority_frac < 0.05:
        verdict = ("severely imbalanced: the minority class is <5% of graphs; "
                   "a model trained here alone cannot learn it")
    elif minority_frac < 0.20:
        verdict = "imbalanced: minority class present but under-represented"
    else:
        verdict = "reasonably balanced"

    return {
        "corpus": corpus,
        "n_graphs": n,
        "label_balance": {
            "matching_wins": matching,
            "pauli_wins": pauli,
            "ties": ties,
            "matching_win_frac": round(matching / n, 4) if n else 0.0,
            "minority_frac": minority_frac,
            "imbalance_ratio": imbalance_ratio,
            "verdict": verdict,
        },
        "win_rate_by_n": by_n,
        "degenerate_features": degenerate,
    }


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _savefig(fig, out_dir: Path, name: str):
    """Save as a vector PDF (IEEE style)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf)
    log.info("wrote %s", pdf)


def plot_label_distribution(corpora, out_dir: Path):
    """Signed-log delta_cx distribution per corpus (single-column figure).

    Signed-log keeps both the near-zero ties and the huge tails readable.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def slog(x):
        return np.sign(x) * np.log10(np.abs(x) + 1)

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.6))
    for name, df in corpora.items():
        ax.hist(slog(df.delta_cx), bins=60, alpha=0.5,
                color=CORPUS_COLOR[name], label=CORPUS_LABEL[name], density=True)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("signed log10(|delta_cx| + 1)   (>0: matching cheaper)")
    ax.set_ylabel("density")
    ax.set_title("Cost-gap distribution per corpus")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_dir, "label_distribution")
    plt.close(fig)


def plot_win_rate_vs_n(corpora, out_dir: Path):
    """Matching-win rate as a function of graph size (single-column figure)."""
    import matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.6))
    for name, df in corpora.items():
        g = df.groupby("n_vertices")["delta_cx"].apply(lambda s: (s > 0).mean())
        ax.plot(g.index, g.values, "o-", color=CORPUS_COLOR[name],
                label=CORPUS_LABEL[name])
    ax.axhline(0.5, color="k", lw=1, ls=":")
    ax.set_xscale("log", base=2)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("vertices n (log2)")
    ax.set_ylabel("fraction where matching wins")
    ax.set_title("Matching-win rate vs graph size")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_dir, "matching_win_rate")
    plt.close(fig)


def _model_features() -> list:
    """Features fed to the importance model: all except the degenerate ones
    (is_bipartite, is_tree, is_regular) and n_vertices (a design axis, not a
    graph-structure feature). The redundant scale features (edge_count,
    cycle_count, chromatic_index_lower_bound) are no longer extracted at all, so
    they need not be dropped here."""
    drop = {"is_bipartite", "is_tree", "is_regular", "n_vertices"}
    return [c for c in FEATURE_COLS if c not in drop]


def feature_importance(df) -> "dict[str, float]":
    """Permutation importance of each feature for predicting delta_cx.

    Features are z-scored (importance reflects structure, not numeric scale) and
    the redundant raw-scale features are excluded up front (see _model_features:
    edge_count and cycle_count are dropped in favour of edge_density, etc.).
    Removing redundancy matters because permutation importance is unreliable
    under collinearity: given two features that are scalar multiples (edge_count
    vs edge_density for a fixed n) a tree splits arbitrarily on one and the
    other spuriously looks unimportant. A RandomForest is fit and each feature's
    importance is the mean drop in held-out R^2 when it is shuffled (sklearn
    permutation_importance). Returns {feature: importance}, summing to 1.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    cols = _model_features()
    X = df[[f"{c}_z" for c in cols]].to_numpy(dtype=float)  # standardized
    y = df["delta_cx"].to_numpy(dtype=float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                               random_state=0, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0,
                                n_jobs=-1)
    imp = np.clip(pi.importances_mean, 0, None)
    total = imp.sum() or 1.0
    return {c: float(v / total) for c, v in zip(cols, imp)}


def plot_feature_importance(corpora, out_dir: Path) -> dict:
    """Permutation feature importance for delta_cx, one panel per corpus.

    Returns {"importance": {corpus: {feat: imp}}, "order": {corpus: [feats]}}
    so the downstream plots can order features by importance.
    """
    import matplotlib.pyplot as plt

    imps = {name: feature_importance(df) for name, df in corpora.items()}
    order = {name: sorted(imp, key=imp.get, reverse=True)
             for name, imp in imps.items()}

    # Each panel is ordered by its OWN importance (most important on top), so
    # the feature order differs between panels; no sharey (that would force a
    # single shared order and mislabel the bars).
    n = len(corpora)
    fig, axes = plt.subplots(1, n, figsize=(IEEE_PAGE_WIDTH, 3.4))
    axes = axes if n > 1 else [axes]
    for ax, (name, imp) in zip(axes, imps.items()):
        feats = sorted(imp, key=imp.get)  # ascending -> most important on top
        ax.barh(feats, [imp[f] for f in feats], color=CORPUS_COLOR[name])
        ax.set_title(CORPUS_LABEL[name])
        ax.set_xlabel("permutation importance")
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("Feature importance for delta_cx (per corpus)", y=1.01)
    fig.tight_layout()
    _savefig(fig, out_dir, "feature_importance")
    plt.close(fig)
    return {"importance": imps, "order": order}


def plot_feature_distributions(corpora, out_dir: Path, top_feats: list):
    """Violins of ALL modeled features per corpus, ordered by importance.

    Features are standardized on the POOLED data so all corpora share a common
    scale and their differences stay visible (per-corpus z-scoring would
    collapse each to mean 0). top_feats sets the panel order (most important
    first); any modeled feature not in it is appended.
    """
    import math
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    modeled = _model_features()
    feats = [f for f in top_feats if f in modeled]
    feats += [f for f in modeled if f not in feats]  # append the rest

    both = pd.concat(corpora.values(), ignore_index=True)
    for f in feats:  # pooled z-score
        v = both[f].to_numpy(dtype=float)
        sd = v.std()
        both[f + "_pz"] = (v - v.mean()) / (sd if sd > 0 else 1.0)

    palette = {name: CORPUS_COLOR[name] for name in corpora}
    ncol = 5
    nrow = math.ceil(len(feats) / ncol)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(IEEE_PAGE_WIDTH, 1.9 * nrow))
    axes = axes.ravel()
    for ax, f in zip(axes, feats):
        sns.violinplot(data=both, x="corpus", y=f + "_pz", ax=ax, hue="corpus",
                       order=list(corpora), palette=palette, legend=False,
                       cut=0)
        ax.set_title(f)
        ax.set_xlabel("")
        ax.set_ylabel("z-score")
        ax.set_xticklabels([CORPUS_LABEL[c] for c in corpora], rotation=30)
        ax.tick_params(axis="x", labelsize=5)
    for ax in axes[len(feats):]:  # hide any empty panels
        ax.set_visible(False)
    fig.suptitle("Feature distributions, standardized (ordered by importance)",
                 y=1.005)
    fig.tight_layout()
    _savefig(fig, out_dir, "feature_distributions")
    plt.close(fig)


def _cluster_order(df, cols) -> list:
    """Order features by hierarchical clustering on |correlation|, so correlated
    features sit adjacent (distance = 1 - |corr|, average linkage)."""
    import numpy as np
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    c = df[cols].corr().abs().to_numpy()
    dist = 1.0 - c
    np.fill_diagonal(dist, 0.0)
    idx = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    return [cols[i] for i in idx]


def plot_correlation_heatmaps(corpora, out_dir: Path, top_feats: list):
    """One feature-correlation heatmap per corpus over all modeled features.

    Each panel is ordered by hierarchical clustering on ITS OWN correlation
    matrix, so each corpus shows its own tightest correlation blocks (the
    orders may therefore differ between panels).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cols = _model_features()
    n = len(corpora)
    fig, axes = plt.subplots(1, n, figsize=(IEEE_PAGE_WIDTH, 2.6))
    axes = axes if n > 1 else [axes]
    mesh = None
    for i, (ax, (name, df)) in enumerate(zip(axes, corpora.items())):
        ordered = _cluster_order(df, cols)  # independent per corpus
        corr = df[ordered].corr()
        # cbar=False on every panel so all panels keep the same size; one
        # shared colorbar is added on its own axis afterwards.
        # y-labels shown on every panel: each has its own clustering order, so
        # the row features differ between panels.
        sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    square=True, cbar=False, xticklabels=True,
                    yticklabels=True)
        mesh = ax.collections[0]
        ax.set_title(CORPUS_LABEL[name])
        ax.tick_params(labelsize=4)
    # Reserve right margin for a shared colorbar, so no panel loses width.
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    cax = fig.add_axes([0.915, 0.25, 0.012, 0.5])
    fig.colorbar(mesh, cax=cax)
    cax.tick_params(labelsize=5)
    _savefig(fig, out_dir, "correlation_heatmaps")
    plt.close(fig)


def plot_feature_vs_target(corpora, out_dir: Path, top_feats: list):
    """Scatter of every modeled feature vs delta_cx, one cell per (corpus,
    feature). Layout is corpora x features (rows = corpora, columns = features
    in importance order) so the grid is wide, filling a landscape slide.

    Each feature axis is pooled-standardized (z-score across all corpora) so
    each column shares a common x-scale; delta_cx uses a signed-log so both the
    ties and the large tails stay visible.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def slog(x):
        return np.sign(x) * np.log10(np.abs(x) + 1)

    modeled = _model_features()
    feats = [f for f in top_feats if f in modeled]
    feats += [f for f in modeled if f not in feats]

    both = pd.concat(corpora.values(), ignore_index=True)
    zmean = {f: both[f].mean() for f in feats}
    zstd = {f: (both[f].std() or 1.0) for f in feats}
    names = list(corpora)

    nrow, ncol = len(names), len(feats)          # corpora x features (wide)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(1.35 * ncol, 1.55 * nrow),
                             sharex="col", sharey="row", squeeze=False)
    for i, name in enumerate(names):
        df = corpora[name]
        for j, f in enumerate(feats):
            ax = axes[i][j]
            xz = (df[f] - zmean[f]) / zstd[f]
            ax.scatter(xz, slog(df.delta_cx), s=4, alpha=0.35,
                       marker=CORPUS_MARKER[name], color=CORPUS_COLOR[name],
                       edgecolors="none", rasterized=True)
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(f, fontsize=7, rotation=30, ha="left")
            if j == 0:
                ax.set_ylabel(CORPUS_LABEL[name], fontsize=8)
            if i == nrow - 1:
                ax.set_xlabel("z", fontsize=6)
    fig.suptitle("delta_cx (signed-log) vs each feature, per corpus", y=1.005)
    fig.tight_layout()
    _savefig(fig, out_dir, "feature_vs_target")
    plt.close(fig)


def plot_pca(corpora, out_dir: Path):
    """PCA of the feature space, one panel per corpus.

    The projection is fit once on the source corpora (Erdős-Rényi + structured,
    the full space) so all panels share the same axes and are comparable, then
    each corpus is drawn in its own panel.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cols = _model_features()
    sources = {k: v for k, v in corpora.items() if k != "hybrid"}
    both = pd.concat(sources.values(), ignore_index=True)
    scaler = StandardScaler().fit(both[cols].to_numpy(dtype=float))
    pca = PCA(n_components=2).fit(scaler.transform(
        both[cols].to_numpy(dtype=float)))
    ev = pca.explained_variance_ratio_

    def project(df):
        return pca.transform(scaler.transform(df[cols].to_numpy(dtype=float)))

    projs = {name: project(df) for name, df in corpora.items()}
    # Shared axis limits so the panels are directly comparable.
    allZ = [z for z in projs.values()]
    x_all = [v for z in allZ for v in z[:, 0]]
    y_all = [v for z in allZ for v in z[:, 1]]
    xlim = (min(x_all), max(x_all))
    ylim = (min(y_all), max(y_all))

    n = len(corpora)
    fig, axes = plt.subplots(1, n, figsize=(IEEE_PAGE_WIDTH, 2.6),
                             sharex=True, sharey=True)
    axes = axes if n > 1 else [axes]
    for ax, (name, Z) in zip(axes, projs.items()):
        ax.scatter(Z[:, 0], Z[:, 1], s=6, marker=CORPUS_MARKER[name],
                   alpha=0.4, color=CORPUS_COLOR[name], edgecolors="none",
                   rasterized=True)
        ax.set_title(CORPUS_LABEL[name])
        ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    axes[0].set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    fig.suptitle("PCA of graph-feature space", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir, "pca")
    plt.close(fig)
    return {"pc1_var": round(float(ev[0]), 3), "pc2_var": round(float(ev[1]), 3)}


def run(data_dir: Path, plots_dir: Path) -> dict:
    """Load all corpora, write all plots, return the combined balance report."""
    set_ieee_style()
    corpora = {name: standardize(load_corpus(data_dir, name))
               for name in CORPUS_NAMES}
    log.info("loaded %s", {k: len(v) for k, v in corpora.items()})

    # Feature importance decides which features get the detailed treatment, so
    # the distribution/scatter plots are data-driven, not hand-picked.
    imp = plot_feature_importance(corpora, plots_dir)
    # Top features to highlight: interleave each corpus's ranking, dedup.
    top = []
    for tup in zip(*[imp["order"][c] for c in CORPUS_NAMES]):
        for f in tup:
            if f not in top:
                top.append(f)

    plot_label_distribution(corpora, plots_dir)
    plot_win_rate_vs_n(corpora, plots_dir)
    plot_feature_distributions(corpora, plots_dir, top)
    plot_correlation_heatmaps(corpora, plots_dir, top)
    plot_feature_vs_target(corpora, plots_dir, top)
    pca_info = plot_pca(corpora, plots_dir)

    report = {name: balance_report(df, name) for name, df in corpora.items()}
    report["feature_importance"] = imp["importance"]
    report["top_features"] = top[:6]
    report["pca"] = pca_info
    report["plots_dir"] = str(plots_dir)
    (plots_dir / "balance_report.json").write_text(json.dumps(report, indent=2))
    return report
