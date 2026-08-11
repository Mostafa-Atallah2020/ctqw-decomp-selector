"""Plots built from the saved model CSVs, so figures can be regenerated without
refitting. All use the IEEE style shared with the dataset analysis plots.

Figures: training_curve, roc_pr_curves, confusion_matrix, calibration_curve,
coefficients, decision_boundaries.

Run via scripts/train_logistic.py (or call plot_all on an output dir directly).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless, must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# Reuse the analysis module's IEEE style and per-dataset colors.
from analysis.dataset_analysis import (
    DATASET_COLOR,
    DATASET_LABEL,
    IEEE_PAGE_WIDTH,
    set_ieee_style,
)

log = logging.getLogger("model")

# Per-iteration metrics. ROC/PR-AUC excluded (their own figure roc_pr_curves.pdf).
CURVE_METRICS = [
    ("loss", "log-loss"),
    ("accuracy", "accuracy"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("specificity", "specificity"),
    ("npv", "NPV"),
    ("f1", "F1"),
    ("mcc", "MCC"),
    ("cohen_kappa", "Cohen's kappa"),
    ("savings", "CX savings captured"),
]


def plot_training_curve(hist_csv: Path, out_dir: Path) -> None:
    """Train vs validation LOG-LOSS across epochs, one panel per dataset (loss is
    the only honest convergence curve, other metrics sit near their ceiling from
    iteration 1). Solid = train, dashed = validation. Each panel has its own
    y-axis scaled to its loss range."""
    set_ieee_style()
    hist = pd.read_csv(hist_csv)
    datasets = [c for c in ["er", "structured", "balanced"]
               if c in set(hist.dataset)]

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(2.6 * len(datasets), 2.6),
                             squeeze=False, sharex=True)
    for j, dataset in enumerate(datasets):
        h = hist[hist.dataset == dataset]
        c = DATASET_COLOR[dataset]
        ax = axes[0][j]
        tr = pd.to_numeric(h["train_loss"], errors="coerce")
        va = pd.to_numeric(h["val_loss"], errors="coerce")
        ax.plot(h.iteration, tr, "-", color=c, lw=1.4)
        ax.plot(h.iteration, va, "--", color=c, lw=1.4)
        # Both axes start at 0: honest loss baseline and iteration-0 origin.
        ymax = float(pd.concat([tr, va]).max())
        ax.set_xlim(0, float(h.iteration.max()))
        ax.set_ylim(0, ymax * 1.08)
        ax.set_title(DATASET_LABEL[dataset])
        ax.set_xlabel("iteration")
        ax.set_ylabel("log-loss")

    handles = [Line2D([], [], color="0.3", lw=1.4, ls="-", label="train"),
               Line2D([], [], color="0.3", lw=1.4, ls="--", label="validation")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Logistic regression: training vs validation log-loss", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.pdf", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def _datasets_in(df):
    return [c for c in ["er", "structured", "balanced"] if c in set(df.dataset)]


def plot_roc_pr(preds_csv: Path, out_dir: Path) -> None:
    """ROC and Precision-Recall curves on the held-out test set, one line per
    dataset (PR is the more honest view for the imbalanced datasets)."""
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    set_ieee_style()
    preds = pd.read_csv(preds_csv)
    datasets = _datasets_in(preds)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(IEEE_PAGE_WIDTH, 3.0))
    for dataset in datasets:
        d = preds[preds.dataset == dataset]
        y, p = d.y_true.to_numpy(), d.proba.to_numpy()
        c = DATASET_COLOR[dataset]
        if len(np.unique(y)) < 2:                  # no positives in test
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr, color=c, lw=1.2,
                    label=f"{DATASET_LABEL[dataset]} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax_pr.plot(rec, prec, color=c, lw=1.2,
                   label=f"{DATASET_LABEL[dataset]} (AP {average_precision_score(y, p):.3f})")
        ax_pr.axhline(y.mean(), color=c, lw=0.6, ls=":")  # no-skill baseline
    ax_roc.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--")
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate",
               title="ROC curve")
    ax_pr.set(xlabel="recall", ylabel="precision", title="Precision-Recall")
    # Legends below each panel (labels differ per panel).
    ax_roc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=1, frameon=False, fontsize=6)
    ax_pr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                 ncol=1, frameon=False, fontsize=6)
    fig.suptitle("Held-out test: ROC and PR curves", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.pdf", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_confusion(preds_csv: Path, out_dir: Path) -> None:
    """Confusion matrix on the held-out test set (pooled over sizes), per dataset."""
    from sklearn.metrics import confusion_matrix

    set_ieee_style()
    preds = pd.read_csv(preds_csv)
    datasets = _datasets_in(preds)
    labels = ["Pauli", "matching"]
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(2.4 * len(datasets), 2.6), squeeze=False)
    for j, dataset in enumerate(datasets):
        d = preds[preds.dataset == dataset]
        pred = (d.proba.to_numpy() >= 0.5).astype(int)
        cm = confusion_matrix(d.y_true.to_numpy(), pred, labels=[0, 1])
        ax = axes[0][j]
        ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for cc in range(2):
                ax.text(cc, r, f"{cm[r, cc]}", ha="center", va="center",
                        color="white" if cm[r, cc] > cm.max() / 2 else "black",
                        fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
        ax.set_xlabel("predicted")
        if j == 0:
            ax.set_ylabel("actual")
        ax.set_title(DATASET_LABEL[dataset])
    fig.suptitle("Held-out test: confusion matrices", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.pdf", bbox_inches="tight",
                pad_inches=0.3)
    plt.close(fig)


def plot_calibration(preds_csv: Path, out_dir: Path) -> None:
    """Reliability diagram: predicted probability vs observed frequency."""
    from sklearn.calibration import calibration_curve

    set_ieee_style()
    preds = pd.read_csv(preds_csv)
    datasets = _datasets_in(preds)
    fig, ax = plt.subplots(figsize=(IEEE_PAGE_WIDTH / 2, 3.2))
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--", label="perfect")
    for dataset in datasets:
        d = preds[preds.dataset == dataset]
        y, p = d.y_true.to_numpy(), d.proba.to_numpy()
        if len(np.unique(y)) < 2:
            continue
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "-o", ms=3, color=DATASET_COLOR[dataset],
                lw=1.1, label=DATASET_LABEL[dataset])
    ax.set(xlabel="mean predicted probability", ylabel="observed frequency",
           title="Calibration (reliability)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.pdf", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_coefficients(coef_csv: Path, out_dir: Path) -> None:
    """Learned logistic coefficients on standardized features, per dataset. A
    positive weight pushes toward 'matching wins'."""
    set_ieee_style()
    coef = pd.read_csv(coef_csv)
    coef = coef[coef.feature != "(intercept)"]     # drop bias term
    datasets = _datasets_in(coef)
    feats = list(dict.fromkeys(coef.feature))
    y = np.arange(len(feats))
    h = 0.8 / max(len(datasets), 1)
    fig, ax = plt.subplots(figsize=(IEEE_PAGE_WIDTH, 0.34 * len(feats) + 1))
    for k, dataset in enumerate(datasets):
        d = coef[coef.dataset == dataset].set_index("feature").reindex(feats)
        ax.barh(y + k * h, d.coef.to_numpy(), h, color=DATASET_COLOR[dataset],
                label=DATASET_LABEL[dataset])
    ax.axvline(0, color="0.5", lw=0.6)
    ax.set_yticks(y + h * (len(datasets) - 1) / 2)
    ax.set_yticklabels(feats)
    ax.set_xlabel("logistic coefficient (standardized features)")
    ax.set_title("Learned weights: what drives 'matching wins'", pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "coefficients.pdf", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


# Feature pairs for the boundary plots, one per correlation band. Each entry is
# (feature_x, feature_y, label, logx, jitter).
BOUNDARY_PAIRS = [
    ("avg_degree", "max_degree", "correlated (r=+0.99)", False, False),
    ("degree_variance", "diameter", "anti-correlated (r=-0.49)", False, False),
    ("max_matching_size", "avg_clustering", "uncorrelated (r=+0.00)", False, True),
]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _boundary_panel(ax, X2, y, kx, ky, mdl, region_cmap, logx=False,
                    jitter=False):
    """Draw one 2D decision-boundary panel for the REAL trained model.

    `mdl` carries the fitted per-dataset logistic model (coef, scaler mean/scale,
    intercept, median feature vector). The boundary is its P(matching)=0.5 contour
    over the (kx, ky) plane with the other features at their median -- a projection
    of the deployed model, NOT a refit. `logx` only sets the x display scale.
    """
    if len(np.unique(y)) < 2:                     # single class (e.g. ER)
        ax.scatter(X2[:, 0], X2[:, 1], s=4, alpha=0.3, color="tab:red",
                   edgecolors="none", rasterized=True)
        if logx:
            ax.set_xscale("log")
        return False

    coef, mean, scale = mdl["coef"], mdl["mean"], mdl["scale"]
    b0, med = mdl["intercept"], mdl["median"].copy()
    # Full-model log-odds as a function of the raw feature vector row.
    def logodds(rows):
        return b0 + ((rows - mean) / scale) @ coef

    # Zoom OUT (35% past the data) so edge boundaries come into view.
    def padded(v, frac=0.35):
        lo, hi = v.min(), v.max()
        m = (hi - lo) * frac or 1.0
        return lo - m, hi + m
    x0, x1 = padded(X2[:, 0])
    y0, y1 = padded(X2[:, 1])
    # Grid wider than the visible window so the contour stays continuous past the
    # data region (then clipped by set_lim).
    gxr, gyr = x1 - x0, y1 - y0
    xs = np.linspace(x0 - 0.5 * gxr, x1 + 0.5 * gxr, 400)
    ys = np.linspace(y0 - 0.5 * gyr, y1 + 0.5 * gyr, 400)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.tile(med, (gx.size, 1))             # other features at median
    grid[:, kx] = gx.ravel()
    grid[:, ky] = gy.ravel()
    zz = _sigmoid(logodds(grid)).reshape(gx.shape)

    ax.contourf(gx, gy, zz, levels=[0, 0.5, 1], cmap=region_cmap, alpha=0.5)
    ax.contour(gx, gy, zz, levels=[0.5], colors="k", linewidths=1.4)
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    # Plot the LARGER class first, smaller on top, so a class squeezed into a
    # corner isn't hidden. Jitter discrete axes (display only).
    x_plot, y_plot = X2[:, 0].astype(float), X2[:, 1].astype(float)
    if jitter:
        rng = np.random.RandomState(0)
        for arr in (x_plot, y_plot):
            u = np.unique(arr)
            if u.size <= 20:                      # discrete: spread its bands
                step = np.min(np.diff(u)) if u.size > 1 else 1.0
                arr += rng.uniform(-0.35, 0.35, arr.size) * step
    pos = y == 1
    order = ([(~pos, "tab:red"), (pos, "tab:green")]
             if pos.sum() <= (~pos).sum()
             else [(pos, "tab:green"), (~pos, "tab:red")])
    for mask, color in order:
        ax.scatter(x_plot[mask], y_plot[mask], s=6, alpha=0.4, color=color,
                   edgecolors="none", rasterized=True, zorder=3)
    return True


def _load_models(coef_csv: Path, feats):
    """Reconstruct each dataset's trained logistic model from coefficients.csv:
    {dataset: {coef, mean, scale, intercept}} on feature order `feats`."""
    c = pd.read_csv(coef_csv)
    models = {}
    for dataset, g in c.groupby("dataset"):
        g = g.set_index("feature")
        b0 = float(g.loc["(intercept)", "coef"])
        gf = g.reindex(feats)
        models[dataset] = {
            "coef": gf["coef"].to_numpy(float),
            "mean": gf["mean"].to_numpy(float),
            "scale": gf["scale"].to_numpy(float),
            "intercept": b0,
        }
    return models


def plot_decision_boundaries(reg_csv: Path, out_dir: Path) -> None:
    """The REAL trained model's delta_cx = 0 boundary, projected onto
    correlation-chosen feature pairs.

    Rows = BOUNDARY_PAIRS, columns = datasets. Each panel draws the per-dataset
    logistic model's P(matching)=0.5 contour over its two features, others at
    median (NOT a refit, reconstructed from coefficients.csv). Points: green =
    matching wins, red = Pauli wins.
    """
    set_ieee_style()
    df = pd.read_csv(reg_csv)
    datasets = _datasets_in(df)
    feats = [c for c in df.columns if c not in ("dataset", "delta_cx")]
    models = _load_models(out_dir / "coefficients.csv", feats)
    region_cmap = ListedColormap(["#f7d3d3", "#d3f0d8"])   # red / green

    nrow, ncol = len(BOUNDARY_PAIRS), len(datasets)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow),
                             squeeze=False)
    for i, (fx, fy, rel, logx, jitter) in enumerate(BOUNDARY_PAIRS):
        kx, ky = feats.index(fx), feats.index(fy)
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            d = df[(df.dataset == dataset) & (df.delta_cx != 0)]
            X2 = d[[fx, fy]].to_numpy(float)
            y = (d.delta_cx.to_numpy() > 0).astype(int)
            mdl = dict(models[dataset])
            mdl["median"] = d[feats].median().to_numpy(float)
            ok = _boundary_panel(ax, X2, y, kx, ky, mdl, region_cmap,
                                 logx=logx, jitter=jitter)
            if i == 0:
                ax.set_title(DATASET_LABEL[dataset]
                             + ("" if ok else " (one class)"), fontsize=9)
            ax.set_xlabel(fx + (" (log)" if logx else ""), fontsize=7)
            ax.set_ylabel(fy, fontsize=7)
            ax.tick_params(labelsize=6)
        axes[i][0].annotate(rel, xy=(-0.42, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center", fontsize=8)

    handles = [
        Line2D([], [], marker="o", ls="", color="tab:green",
               label=r"matching wins ($\Delta_{CX}>0$)"),
        Line2D([], [], marker="o", ls="", color="tab:red",
               label=r"Pauli wins ($\Delta_{CX}<0$)"),
        Line2D([], [], color="k", lw=1.4, label=r"$\Delta_{CX}=0$ boundary")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Trained logistic model's decision boundary, projected onto the "
                 "best-separating pair per correlation band (others at median)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "decision_boundaries.pdf", bbox_inches="tight",
                pad_inches=0.3)
    plt.close(fig)


def plot_all(out_dir: Path) -> None:
    """Build every plot from the CSVs already written in out_dir."""
    hist = out_dir / "training_history.csv"
    preds = out_dir / "test_predictions.csv"
    coef = out_dir / "coefficients.csv"
    reg = out_dir / "feature_regressions.csv"   # feeds decision-boundary plot
    if hist.exists():
        plot_training_curve(hist, out_dir)
    if preds.exists():
        plot_roc_pr(preds, out_dir)
        plot_confusion(preds, out_dir)
        plot_calibration(preds, out_dir)
    if coef.exists():
        plot_coefficients(coef, out_dir)
    if reg.exists():
        plot_decision_boundaries(reg, out_dir)
    log.info("wrote model plots to %s", out_dir)
