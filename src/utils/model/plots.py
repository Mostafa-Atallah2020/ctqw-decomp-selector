"""Plots built from the saved model CSVs.

Kept separate from training so the figures can be regenerated from the CSVs
alone, without refitting. All figures use the IEEE style shared with the corpus
analysis plots. Grid figures lay out rows = metrics/pairs, columns = corpora.

  - training_curve.pdf       : per-iteration TRAIN vs VALIDATION progress, a grid
                               of rows = metrics (loss, accuracy, precision,
                               recall, specificity, NPV, F1, MCC, Cohen's kappa,
                               savings), columns = corpora. ROC/PR-AUC are on
                               their own figure. Final TEST numbers are in
                               metrics.csv, not here.
  - roc_pr_curves.pdf        : ROC and Precision-Recall curves, per corpus.
  - confusion_matrix.pdf     : confusion matrices, one per corpus.
  - calibration_curve.pdf    : reliability diagram, per corpus.
  - coefficients.pdf         : learned logistic weights, per corpus.
  - decision_boundaries.pdf  : 2D delta_cx = 0 boundaries over correlation-chosen
                               feature pairs (correlated / anti / uncorrelated),
                               rows = pairs, columns = corpora.

Run via scripts/train_model.py (or call plot_all on an output dir directly).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# Reuse the analysis module's IEEE style and per-corpus colors.
from utils.analysis.corpus_analysis import (
    CORPUS_COLOR,
    CORPUS_LABEL,
    IEEE_PAGE_WIDTH,
    set_ieee_style,
)

log = logging.getLogger("model")

# Per-iteration metrics shown on the training curve. ROC/PR-AUC are excluded:
# they have their own figure (roc_pr_curves.pdf).
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
    """Train vs validation progress across epochs, laid out as a grid of
    rows = metrics, columns = corpora (side by side). Solid = train, dashed =
    validation; each column uses its corpus color."""
    set_ieee_style()
    hist = pd.read_csv(hist_csv)
    corpora = [c for c in ["er", "structured", "hybrid"]
               if c in set(hist.corpus)]
    metrics = [(k, lab) for k, lab in CURVE_METRICS
               if f"train_{k}" in hist.columns]

    nrow, ncol = len(metrics), len(corpora)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(2.4 * ncol, 1.35 * nrow),
                             squeeze=False, sharex=True)
    for j, corpus in enumerate(corpora):
        h = hist[hist.corpus == corpus]
        c = CORPUS_COLOR[corpus]
        for i, (key, label) in enumerate(metrics):
            ax = axes[i][j]
            tr = pd.to_numeric(h[f"train_{key}"], errors="coerce")
            va = pd.to_numeric(h[f"val_{key}"], errors="coerce")
            ax.plot(h.iteration, tr, "-", color=c, lw=1.0)
            ax.plot(h.iteration, va, "--", color=c, lw=1.0)
            if i == 0:                             # corpus name atop each column
                ax.set_title(CORPUS_LABEL[corpus])
            if j == 0:                             # metric name on left column
                ax.set_ylabel(label)
            if i == nrow - 1:
                ax.set_xlabel("iteration")

    handles = [Line2D([], [], color="0.3", lw=1.4, ls="-", label="train"),
               Line2D([], [], color="0.3", lw=1.4, ls="--", label="validation")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Logistic regression: training vs validation", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.pdf", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def _corpora_in(df):
    return [c for c in ["er", "structured", "hybrid"] if c in set(df.corpus)]


def plot_roc_pr(preds_csv: Path, out_dir: Path) -> None:
    """ROC and Precision-Recall curves on the held-out test set, one line per
    corpus. PR is the more honest view for the imbalanced corpora."""
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    set_ieee_style()
    preds = pd.read_csv(preds_csv)
    corpora = _corpora_in(preds)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(IEEE_PAGE_WIDTH, 3.0))
    for corpus in corpora:
        d = preds[preds.corpus == corpus]
        y, p = d.y_true.to_numpy(), d.proba.to_numpy()
        c = CORPUS_COLOR[corpus]
        if len(np.unique(y)) < 2:                  # e.g. no positives in test
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr, color=c, lw=1.2,
                    label=f"{CORPUS_LABEL[corpus]} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax_pr.plot(rec, prec, color=c, lw=1.2,
                   label=f"{CORPUS_LABEL[corpus]} (AP {average_precision_score(y, p):.3f})")
        ax_pr.axhline(y.mean(), color=c, lw=0.6, ls=":")  # no-skill baseline
    ax_roc.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--")
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate",
               title="ROC curve")
    ax_pr.set(xlabel="recall", ylabel="precision", title="Precision-Recall")
    # Legends below each panel (AUC/AP labels differ per panel).
    ax_roc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=1, frameon=False, fontsize=6)
    ax_pr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                 ncol=1, frameon=False, fontsize=6)
    fig.suptitle("Held-out test: ROC and PR curves", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.pdf", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_confusion(preds_csv: Path, out_dir: Path) -> None:
    """Confusion matrix on the held-out test set (pooled over sizes), one panel
    per corpus."""
    from sklearn.metrics import confusion_matrix

    set_ieee_style()
    preds = pd.read_csv(preds_csv)
    corpora = _corpora_in(preds)
    labels = ["Pauli", "matching"]
    fig, axes = plt.subplots(1, len(corpora),
                             figsize=(2.4 * len(corpora), 2.6), squeeze=False)
    for j, corpus in enumerate(corpora):
        d = preds[preds.corpus == corpus]
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
        ax.set_title(CORPUS_LABEL[corpus])
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
    corpora = _corpora_in(preds)
    fig, ax = plt.subplots(figsize=(IEEE_PAGE_WIDTH / 2, 3.2))
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--", label="perfect")
    for corpus in corpora:
        d = preds[preds.corpus == corpus]
        y, p = d.y_true.to_numpy(), d.proba.to_numpy()
        if len(np.unique(y)) < 2:
            continue
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "-o", ms=3, color=CORPUS_COLOR[corpus],
                lw=1.1, label=CORPUS_LABEL[corpus])
    ax.set(xlabel="mean predicted probability", ylabel="observed frequency",
           title="Calibration (reliability)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.pdf", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_coefficients(coef_csv: Path, out_dir: Path) -> None:
    """Learned logistic coefficients on standardized features, per corpus. A
    positive weight pushes the prediction toward 'matching wins'."""
    set_ieee_style()
    coef = pd.read_csv(coef_csv)
    coef = coef[coef.feature != "(intercept)"]     # drop the bias term
    corpora = _corpora_in(coef)
    feats = list(dict.fromkeys(coef.feature))
    y = np.arange(len(feats))
    h = 0.8 / max(len(corpora), 1)
    fig, ax = plt.subplots(figsize=(IEEE_PAGE_WIDTH, 0.34 * len(feats) + 1))
    for k, corpus in enumerate(corpora):
        d = coef[coef.corpus == corpus].set_index("feature").reindex(feats)
        ax.barh(y + k * h, d.coef.to_numpy(), h, color=CORPUS_COLOR[corpus],
                label=CORPUS_LABEL[corpus])
    ax.axvline(0, color="0.5", lw=0.6)
    ax.set_yticks(y + h * (len(corpora) - 1) / 2)
    ax.set_yticklabels(feats)
    ax.set_xlabel("logistic coefficient (standardized features)")
    ax.set_title("Learned weights: what drives 'matching wins'", pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "coefficients.pdf", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


# Feature pairs to draw boundaries over, chosen from the correlation heatmap to
# span the range of correlation structure. Each entry is
# (feature_x, feature_y, label, logx): logx log-scales the x-axis where the
# feature spans orders of magnitude, so the classes are not squeezed.
BOUNDARY_PAIRS = [
    # One pair per correlation band (correlated / anti-correlated / uncorrelated),
    # picking the BEST-SEPARATING pair within each band (highest MCC of a
    # 2-feature fit on the hybrid). (feature_x, feature_y, label, logx, jitter)
    ("avg_degree", "max_degree", "correlated (r=+0.99)", False, False),
    ("degree_variance", "diameter", "anti-correlated (r=-0.49)", False, False),
    ("max_matching_size", "avg_clustering", "uncorrelated (r=+0.00)", False, True),
]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _boundary_panel(ax, X2, y, kx, ky, mdl, region_cmap, logx=False,
                    jitter=False):
    """Draw one 2D decision-boundary panel for the REAL trained model.

    `mdl` carries the fitted logistic model per corpus: standardized coefficients
    `coef`, scaler `mean`/`scale` (per feature), `intercept`, and `median` (the
    full feature vector at its per-feature median). The boundary is this model's
    P(matching)=0.5 contour over the (kx, ky) feature plane, with the other 8
    features held at their median -- a true projection of the deployed model, NOT
    a refit. `logx` only sets the x-axis display scale (the model uses raw x).
    """
    if len(np.unique(y)) < 2:                     # single class (e.g. ER)
        ax.scatter(X2[:, 0], X2[:, 1], s=4, alpha=0.3, color="tab:red",
                   edgecolors="none", rasterized=True)
        if logx:
            ax.set_xscale("log")
        return False

    coef, mean, scale = mdl["coef"], mdl["mean"], mdl["scale"]
    b0, med = mdl["intercept"], mdl["median"].copy()
    # Log-odds of the full model as a function of the raw feature vector row.
    def logodds(rows):
        return b0 + ((rows - mean) / scale) @ coef

    # Zoom OUT (35% past the data) so boundaries that sit near the data edge come
    # into view instead of being clipped tight against the axes.
    def padded(v, frac=0.35):
        lo, hi = v.min(), v.max()
        m = (hi - lo) * frac or 1.0
        return lo - m, hi + m
    x0, x1 = padded(X2[:, 0])
    y0, y1 = padded(X2[:, 1])
    # Grid wider still than the visible window so the contour stays continuous
    # even where the boundary leaves the data region (then clipped by set_lim).
    gxr, gyr = x1 - x0, y1 - y0
    xs = np.linspace(x0 - 0.5 * gxr, x1 + 0.5 * gxr, 400)
    ys = np.linspace(y0 - 0.5 * gyr, y1 + 0.5 * gyr, 400)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.tile(med, (gx.size, 1))             # other 8 features at median
    grid[:, kx] = gx.ravel()
    grid[:, ky] = gy.ravel()
    zz = _sigmoid(logodds(grid)).reshape(gx.shape)

    ax.contourf(gx, gy, zz, levels=[0, 0.5, 1], cmap=region_cmap, alpha=0.5)
    ax.contour(gx, gy, zz, levels=[0.5], colors="k", linewidths=1.4)
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    # Plot the LARGER class first and the smaller/compressed class on top, so a
    # class squeezed into a corner (e.g. hybrid matching-wins at low degree)
    # isn't hidden under the other. Green (matching) drawn last where it is
    # smaller-spread.
    # Jitter whichever axis is discrete (few distinct values), for display only.
    x_plot, y_plot = X2[:, 0].astype(float), X2[:, 1].astype(float)
    if jitter:
        rng = np.random.RandomState(0)
        for arr in (x_plot, y_plot):
            u = np.unique(arr)
            if u.size <= 20:                      # discrete -> spread its bands
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
    """Reconstruct each corpus's trained logistic model from coefficients.csv:
    returns {corpus: {coef, mean, scale, intercept}} on the feature order `feats`."""
    c = pd.read_csv(coef_csv)
    models = {}
    for corpus, g in c.groupby("corpus"):
        g = g.set_index("feature")
        b0 = float(g.loc["(intercept)", "coef"])
        gf = g.reindex(feats)
        models[corpus] = {
            "coef": gf["coef"].to_numpy(float),
            "mean": gf["mean"].to_numpy(float),
            "scale": gf["scale"].to_numpy(float),
            "intercept": b0,
        }
    return models


def plot_decision_boundaries(reg_csv: Path, out_dir: Path) -> None:
    """The REAL trained model's delta_cx = 0 decision boundary, projected onto
    correlation-chosen feature pairs.

    Rows are the feature pairs in BOUNDARY_PAIRS (correlated / anti-correlated /
    uncorrelated, per the correlation heatmap); columns are the corpora. Each
    panel draws the actual per-corpus logistic model's P(matching) = 0.5 contour
    over its two features, holding the other 8 at their median -- NOT a refit.
    The model is reconstructed from coefficients.csv (coef + scaler mean/scale).
    Points: green = +delta (matching wins), red = -delta (Pauli wins).
    """
    set_ieee_style()
    df = pd.read_csv(reg_csv)
    corpora = _corpora_in(df)
    feats = [c for c in df.columns if c not in ("corpus", "delta_cx")]
    models = _load_models(out_dir / "coefficients.csv", feats)
    region_cmap = ListedColormap(["#f7d3d3", "#d3f0d8"])   # red-ish / green-ish

    nrow, ncol = len(BOUNDARY_PAIRS), len(corpora)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow),
                             squeeze=False)
    for i, (fx, fy, rel, logx, jitter) in enumerate(BOUNDARY_PAIRS):
        kx, ky = feats.index(fx), feats.index(fy)
        for j, corpus in enumerate(corpora):
            ax = axes[i][j]
            d = df[(df.corpus == corpus) & (df.delta_cx != 0)]
            X2 = d[[fx, fy]].to_numpy(float)
            y = (d.delta_cx.to_numpy() > 0).astype(int)
            mdl = dict(models[corpus])
            mdl["median"] = d[feats].median().to_numpy(float)
            ok = _boundary_panel(ax, X2, y, kx, ky, mdl, region_cmap,
                                 logx=logx, jitter=jitter)
            if i == 0:
                ax.set_title(CORPUS_LABEL[corpus]
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
    reg = out_dir / "feature_regressions.csv"   # feeds the decision-boundary plot
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
