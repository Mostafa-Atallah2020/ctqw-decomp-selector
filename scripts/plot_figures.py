#!/usr/bin/env python3
"""Generate figures over the McKay population (data analysis and tuned-ANN evaluation).

Figure sets routed by --fig: analysis, importance, roc-pr, all (default).

Examples:
    python scripts/plot_figures.py --fig analysis
    python scripts/plot_figures.py --fig importance --seeds 0 1 2 --repeats 20
"""
from __future__ import annotations

import argparse
import ast
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from model import evaluation as _bm  # noqa: E402

from analysis.dataset_analysis import (  # noqa: E402
    IEEE_PAGE_WIDTH, set_ieee_style)

# Shared configuration
DECOMP = ["pauli_num_terms", "match_num_matchings"]
SEEDS = [0, 1, 2, 3, 4]

ANALYSIS_OUT = _ROOT / "results" / "analysis"
EVAL_OUT = _ROOT / "results" / "evaluation"

# Class colors: matching-win vs Pauli-win.
C_MATCH, C_PAULI = "#C55A11", "#4C72B0"

# Tuned winner from hp_deep_tune, no random_state so each caller supplies its own.
ANN_KW = dict(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-4,
              learning_rate_init=1e-3, max_iter=1000)

# Compact display names.
PRETTY = {
    "edge_density": "edge density", "avg_degree": "avg degree",
    "max_degree": "max degree", "min_degree": "min degree",
    "degree_variance": "degree variance", "max_matching_size": "max matching size",
    "spectral_gap": "spectral gap", "avg_clustering": "avg clustering",
    "triangle_count": "triangle count", "diameter": "diameter",
    "pauli_num_terms": r"$n_{\mathrm{Pauli}}$",
    "match_num_matchings": r"$n_{\mathrm{match}}$",
    "pauli_match_ratio": r"$n_{\mathrm{Pauli}}/n_{\mathrm{match}}$",
    "pauli_match_diff": r"$n_{\mathrm{Pauli}}-n_{\mathrm{match}}$",
    "pauli_wt_mean": "Pauli weight mean", "pauli_wt_max": "Pauli weight max",
    "pauli_wt_std": "Pauli weight std",
}

# Shorter labels used only by the analysis figures, to preserve their output.
_ANALYSIS_PRETTY = dict(PRETTY)
_ANALYSIS_PRETTY.update({
    "degree_variance": "degree var.", "max_matching_size": "max matching",
    "triangle_count": "triangles",
})


def _load():
    """Load McKay n=8, nonzero-gap graphs with valid decomp counts. y=1 matching wins."""
    df = _bm.load("mckay")
    df = df[(df.n_vertices == 8) & (df.delta_cx != 0)].dropna(subset=DECOMP).copy()
    df["y"] = (df.delta_cx > 0).astype(int)          # 1 = matching wins
    feats = list(_bm.FEATURES) + DECOMP
    return df, feats


# analysis
def _pname(f):
    return _ANALYSIS_PRETTY.get(f, f)


def plot_feature_vs_target(df, feats):
    """Each z-scored feature vs the signed-log cost gap, points colored by class."""
    def slog(x):
        return np.sign(x) * np.log10(np.abs(x) + 1)

    ncol = 4
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 1.7 * nrow),
                             sharey=True, squeeze=False)
    yslog = slog(df.delta_cx.to_numpy())
    for k, f in enumerate(feats):
        ax = axes[k // ncol][k % ncol]
        xz = (df[f] - df[f].mean()) / (df[f].std() or 1.0)
        for cls, col in [(0, C_PAULI), (1, C_MATCH)]:
            m = df.y.to_numpy() == cls
            ax.scatter(xz[m], yslog[m], s=5, alpha=0.35, color=col,
                       rasterized=True, linewidths=0)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_title(_pname(f), fontsize=9)
        ax.tick_params(labelsize=7)
        if k % ncol == 0:
            ax.set_ylabel(r"signed-log $\Delta_{\mathrm{CX}}$", fontsize=8)
        ax.set_xlabel("z-score", fontsize=7)
    for k in range(len(feats), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C_MATCH, label="matching wins"),
                        Patch(color=C_PAULI, label="Pauli wins")],
               loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ANALYSIS_OUT / "feature_vs_target_mckay.pdf", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def _cluster_order(df, cols):
    """Order features by hierarchical clustering on |correlation| (distance 1-|corr|,
    average linkage) so correlated features sit adjacent."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    c = df[cols].corr().abs().to_numpy()
    dist = 1.0 - c
    np.fill_diagonal(dist, 0.0)
    idx = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    return [cols[i] for i in idx]


def plot_correlation_heatmap(df, feats):
    """Feature-feature Pearson correlation, clustered so correlated blocks are visible."""
    ordered = _cluster_order(df, feats)
    corr = df[ordered].corr().to_numpy()
    labels = [_pname(f) for f in ordered]
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered)))
    ax.set_yticks(range(len(ordered)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ANALYSIS_OUT / "correlation_heatmap_mckay.pdf", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def plot_pca(df, feats):
    """Two-component PCA of the standardized feature space, points by class."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(df[feats])
    p = PCA(n_components=2, random_state=0)
    Z = p.fit_transform(X)
    ev = p.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    # Majority (Pauli) first so the rare matching-wins sit on top.
    for cls, col, lab in [(0, C_PAULI, "Pauli wins"),
                          (1, C_MATCH, "matching wins")]:
        m = df.y.to_numpy() == cls
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.4, color=col, label=lab,
                   rasterized=True, linewidths=0)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax.legend(fontsize=7, frameon=False, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(ANALYSIS_OUT / "pca_mckay.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_feature_distributions(df, feats):
    """Class-conditional distribution of each feature, one panel per feature.

    Both classes share raw-count (linear y) and bin edges per feature, so the 25:1
    imbalance reads honestly and every panel measures the same quantity.
    """
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow),
                             squeeze=False)
    n_pauli = int((df.y == 0).sum())
    n_match = int((df.y == 1).sum())
    for k, f in enumerate(feats):
        ax = axes[k // ncol][k % ncol]
        v_all = df[f].to_numpy()
        # shared bin edges so the two classes are directly comparable
        lo, hi = np.nanmin(v_all), np.nanmax(v_all)
        uniq = np.unique(v_all[~np.isnan(v_all)])
        if uniq.size <= 12:                       # small-integer feature
            step = np.min(np.diff(uniq)) if uniq.size > 1 else 1.0
            edges = np.append(uniq - step / 2, uniq[-1] + step / 2)
        else:
            edges = np.linspace(lo, hi, 31)
        # Pauli first (tall), matching on top (thin), both raw linear counts.
        ax.hist(df.loc[df.y == 0, f], bins=edges, color=C_PAULI, alpha=0.8,
                label="Pauli wins")
        ax.hist(df.loc[df.y == 1, f], bins=edges, color=C_MATCH, alpha=0.9,
                label="matching wins")
        ax.set_title(_pname(f), fontsize=11)
        ax.set_ylabel("graph count", fontsize=8)
        ax.tick_params(labelsize=8)
    for k in range(len(feats), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C_PAULI,
                              label=f"Pauli wins (n={n_pauli:,})"),
                        Patch(color=C_MATCH,
                              label=f"matching wins (n={n_match:,})")],
               loc="lower center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Class-conditional feature distributions on the McKay "
                 "population (shared linear count scale, note the 25:1 imbalance)",
                 fontsize=11, y=1.01)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    fig.savefig(ANALYSIS_OUT / "feature_distributions_mckay.pdf",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _tuned_ann_predictions(df, feats):
    """Fit the tuned ANN per seed and pool held-out test predictions (five-seed protocol)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    ys, probas, preds = [], [], []
    per_seed = []                                     # (y, pred) per seed fold
    for seed in SEEDS:
        tr, _v, te = _bm.split(df, seed)
        sc = StandardScaler().fit(tr[feats])
        Xtr, Xte = sc.transform(tr[feats]), sc.transform(te[feats])
        clf = MLPClassifier(random_state=0, **ANN_KW).fit(Xtr, tr.y)
        p = clf.predict_proba(Xte)[:, 1]
        yte = te.y.to_numpy(); pr = (p >= 0.5).astype(int)
        ys.append(yte); probas.append(p); preds.append(pr)
        per_seed.append((yte, pr))
    return (np.concatenate(ys), np.concatenate(probas),
            np.concatenate(preds), per_seed)


def plot_ann_convergence(df, feats, n_epochs=300):
    """Train/val convergence of the tuned ANN on seed 0, one epoch at a time
    (warm_start), plotting log-loss and MCC to confirm it is not overfitting.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import matthews_corrcoef, log_loss

    tr, va, _te = _bm.split(df, 0)
    sc = StandardScaler().fit(tr[feats])
    Xtr, Xva = sc.transform(tr[feats]), sc.transform(va[feats])
    ytr, yva = tr.y.to_numpy(), va.y.to_numpy()

    kw = dict(ANN_KW, random_state=0)
    kw.update(max_iter=1, warm_start=True)
    clf = MLPClassifier(**kw)
    tr_loss, va_loss, tr_mcc, va_mcc = [], [], [], []
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(n_epochs):
            clf.fit(Xtr, ytr)                         # one more epoch per call
            ptr = clf.predict_proba(Xtr)[:, 1]
            pva = clf.predict_proba(Xva)[:, 1]
            tr_loss.append(log_loss(ytr, ptr, labels=[0, 1]))
            va_loss.append(log_loss(yva, pva, labels=[0, 1]))
            tr_mcc.append(matthews_corrcoef(ytr, (ptr >= 0.5).astype(int)))
            va_mcc.append(matthews_corrcoef(yva, (pva >= 0.5).astype(int)))

    epochs = np.arange(1, n_epochs + 1)
    fig, (axl, axm) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    axl.plot(epochs, tr_loss, color=C_PAULI, lw=1.3, label="train")
    axl.plot(epochs, va_loss, color=C_MATCH, lw=1.3, label="validation")
    axl.set_xlabel("epoch"); axl.set_ylabel("log-loss")
    axl.legend(fontsize=8, frameon=False)
    axm.plot(epochs, tr_mcc, color=C_PAULI, lw=1.3, label="train")
    axm.plot(epochs, va_mcc, color=C_MATCH, lw=1.3, label="validation")
    axm.set_xlabel("epoch"); axm.set_ylabel("MCC")
    axm.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ANALYSIS_OUT / "ann_convergence_mckay.pdf", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    return va_mcc[-1]


def plot_pr_curve(y, proba):
    """Precision-Recall curve for the tuned ANN (the honest view under imbalance)."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.plot(rec, prec, color=C_MATCH, lw=1.4)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.text(0.05, 0.08, f"PR-AUC {ap:.3f}", fontsize=8, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(ANALYSIS_OUT / "pr_curve_mckay.pdf", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    return ap


def plot_confusion(per_seed):
    """Confusion matrix for the tuned ANN, each cell the per-seed mean +- std over
    the five held-out test folds. color encodes the mean count."""
    from sklearn.metrics import confusion_matrix
    mats = np.stack([confusion_matrix(yte, pr, labels=[0, 1])
                     for yte, pr in per_seed])         # (n_seeds, 2, 2)
    mean = mats.mean(axis=0)
    std = mats.std(axis=0)
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    ax.imshow(mean, cmap="Oranges")
    labels = ["Pauli", "matching"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    thresh = mean.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mean[i, j]:.0f} $\\pm$ {std[i, j]:.0f}",
                    ha="center", va="center",
                    color="white" if mean[i, j] > thresh else "black",
                    fontsize=9)
    fig.tight_layout()
    fig.savefig(ANALYSIS_OUT / "confusion_mckay.pdf", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    return mean, std


def plot_calibration(y, proba):
    """Reliability diagram (uniform bins) for the tuned ANN, with a histogram of the
    predicted probabilities beneath it showing why most bins are sparse.
    """
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="uniform")
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(3.0, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", color=C_MATCH, lw=1.4, ms=4,
            label="tuned ANN")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    # Log-scaled: the near-zero bin dominates by two orders of magnitude.
    axh.hist(proba, bins=20, range=(0, 1), color=C_MATCH, log=True)
    axh.set_xlabel("predicted matching-probability")
    axh.set_ylabel("count", fontsize=7)
    axh.tick_params(labelsize=6)
    fig.savefig(ANALYSIS_OUT / "calibration_mckay.pdf", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def run_analysis(args) -> int:
    """Data-analysis and tuned-ANN evaluation figures."""
    set_ieee_style()
    df, feats = _load()
    print(f"McKay population: {len(df)} graphs, {int(df.y.sum())} matching-wins, "
          f"{len(feats)} features")
    plot_feature_vs_target(df, feats)
    print("  wrote feature_vs_target_mckay.pdf")
    plot_correlation_heatmap(df, feats)
    print("  wrote correlation_heatmap_mckay.pdf")
    plot_pca(df, feats)
    print("  wrote pca_mckay.pdf")
    plot_feature_distributions(df, feats)
    print("  wrote feature_distributions_mckay.pdf")

    # Tuned-ANN evaluation figures over five held-out folds.
    y, proba, pred, per_seed = _tuned_ann_predictions(df, feats)
    ap = plot_pr_curve(y, proba)
    print(f"  wrote pr_curve_mckay.pdf (AP {ap:.3f})")
    mean, std = plot_confusion(per_seed)
    print(f"  wrote confusion_mckay.pdf (per-seed mean+-std) "
          f"TN {mean[0,0]:.0f}+-{std[0,0]:.0f}, FP {mean[0,1]:.0f}+-{std[0,1]:.0f}, "
          f"FN {mean[1,0]:.0f}+-{std[1,0]:.0f}, TP {mean[1,1]:.0f}+-{std[1,1]:.0f}")
    plot_calibration(y, proba)
    print("  wrote calibration_mckay.pdf")
    va_mcc = plot_ann_convergence(df, feats)
    print(f"  wrote ann_convergence_mckay.pdf (final val MCC {va_mcc:.3f})")
    return 0


# importance
IMP_OUT_CSV = EVAL_OUT / "feature_importance_mckay.csv"
IMP_OUT_PDF = EVAL_OUT / "feature_importance_mckay.pdf"


def run_importance(args) -> int:
    """Permutation feature importance on the tuned ANN."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import matthews_corrcoef, make_scorer

    df = _bm.load("mckay")
    df = df[(df.n_vertices == 8) & (df.delta_cx != 0)].dropna(subset=DECOMP).copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    cols = list(_bm.FEATURES) + DECOMP
    print(f"mckay (Brandon McKay): {len(df)} graphs, {int(df.y.sum())} positives, "
          f"{len(cols)} features ({len(_bm.FEATURES)} topological + {len(DECOMP)} "
          f"decomposition)")

    mcc_scorer = make_scorer(matthews_corrcoef)
    per_seed = []                                # (n_seeds, n_features) importances
    for seed in args.seeds:
        tr, _v, te = _bm.split(df, seed)
        # Pipeline ties the scaler to the ANN so permuting a raw feature scores end to end.
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(random_state=seed, **ANN_KW))
        clf.fit(tr[cols].to_numpy(), tr.y)
        r = permutation_importance(
            clf, te[cols].to_numpy(), te.y.to_numpy(),
            scoring=mcc_scorer, n_repeats=args.repeats, random_state=seed, n_jobs=-1)
        per_seed.append(r.importances_mean)
        print(f"  seed {seed}: fitted and permuted ({args.repeats} repeats)")

    imp = np.array(per_seed)                     # (seeds, features)
    mean = imp.mean(axis=0)
    std = imp.std(axis=0)
    res = (pd.DataFrame({"feature": cols, "importance": mean, "importance_std": std,
                         "group": ["decomposition" if c in DECOMP else "topological"
                                   for c in cols]})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    res.to_csv(IMP_OUT_CSV, index=False)
    print(f"\nwrote {IMP_OUT_CSV}\n")
    print(f"{'feature':<22}{'group':<14}{'importance':>12}")
    for _, row in res.iterrows():
        print(f"  {row.feature:<20}{row.group:<14}{row.importance:>12.4f}")

    # Horizontal bar chart, most important at top, single uniform series.
    ordered = res.iloc[::-1].reset_index(drop=True)   # smallest at bottom of axis
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    y = np.arange(len(ordered))
    ax.barh(y, ordered.importance, xerr=ordered.importance_std,
            color="#C55A11", ecolor="0.4", capsize=1.5, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY.get(f, f) for f in ordered.feature], fontsize=7)
    ax.set_xlabel("permutation importance", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.margins(y=0.01)
    fig.tight_layout()
    fig.savefig(IMP_OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"\nwrote {IMP_OUT_PDF}")
    return 0


# roc-pr
ROC_TUNED = EVAL_OUT / "hp_deep_tune_mckay.csv"
ROC_OUT = EVAL_OUT / "roc_pr_tuned_mckay.pdf"


def _roc_base():
    """Base estimator + needs_scaling per model name, built lazily so the boosting
    imports load only when roc-pr is requested."""
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "logistic": (lambda: LogisticRegression(max_iter=5000,
                                                class_weight="balanced"), True),
        "naive_bayes": (GaussianNB, True),
        "dtree": (lambda: DecisionTreeClassifier(class_weight="balanced",
                                                 random_state=0), False),
        "knn": (KNeighborsClassifier, True),
        "svm_rbf": (lambda: SVC(probability=True, class_weight="balanced",
                                random_state=0), True),
        "mlp": (lambda: MLPClassifier(max_iter=1000, random_state=0), True),
        "rand_forest": (lambda: RandomForestClassifier(class_weight="balanced",
                                                       random_state=0, n_jobs=-1),
                        False),
        "grad_boost": (lambda: GradientBoostingClassifier(random_state=0), False),
        "lightgbm": (lambda: LGBMClassifier(verbose=-1, random_state=0, n_jobs=-1),
                     False),
        "xgboost": (lambda: XGBClassifier(eval_metric="logloss", verbosity=0,
                                          random_state=0, n_jobs=-1), False),
    }


ROC_LABEL = {
    "mlp": "ANN", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "rand_forest": "random forest", "grad_boost": "gradient boosting",
    "svm_rbf": "RBF-SVM", "knn": "k-NN", "dtree": "decision tree",
    "logistic": "logistic", "naive_bayes": "naive Bayes",
}


def _parse_params(s: str) -> dict:
    """Parse the sweep's best_params dict-repr, stripping np.float64/np.int64 wrappers."""
    return ast.literal_eval(s.replace("np.float64(", "(").replace("np.int64(", "("))


def run_roc_pr(args) -> int:
    """ROC/PR curves for every tuned model on the BM population."""
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    from sklearn.preprocessing import StandardScaler

    BASE = _roc_base()

    df = _bm.load("mckay")
    df = df[(df.n_vertices == 8) & (df.delta_cx != 0)].dropna(subset=DECOMP).copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    cols = list(_bm.FEATURES) + DECOMP
    tr, _v, te = _bm.split(df, args.seed)
    print(f"mckay (Brandon McKay): train {len(tr)}, test {len(te)}, "
          f"{int(te.y.sum())} test positives")

    tuned = pd.read_csv(ROC_TUNED)
    curves = []
    for _, row in tuned.iterrows():
        name = row["model"]
        if name not in BASE:
            continue
        make, scale = BASE[name]
        params = _parse_params(row["best_params"])
        Xtr, Xte = tr[cols], te[cols]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        else:
            Xtr, Xte = Xtr.to_numpy(), Xte.to_numpy()
        clf = make()
        clf.set_params(**params)
        clf.fit(Xtr, tr.y)
        proba = clf.predict_proba(Xte)[:, 1]
        y = te.y.to_numpy()
        curves.append((name, roc_auc_score(y, proba),
                       average_precision_score(y, proba), y, proba))
        print(f"  {name:<12} AUC {curves[-1][1]:.3f}  AP {curves[-1][2]:.3f}")

    curves.sort(key=lambda c: -c[1])           # legend ordered by ROC-AUC
    set_ieee_style()
    cmap = plt.get_cmap("tab10")
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(IEEE_PAGE_WIDTH, 3.0))
    for i, (name, auc, ap, y, proba) in enumerate(curves):
        c = cmap(i % 10)
        fpr, tpr, _ = roc_curve(y, proba)
        ax_roc.plot(fpr, tpr, color=c, lw=1.1,
                    label=f"{ROC_LABEL[name]} (AUC {auc:.3f})")
        prec, rec, _ = precision_recall_curve(y, proba)
        ax_pr.plot(rec, prec, color=c, lw=1.1,
                   label=f"{ROC_LABEL[name]} (AP {ap:.3f})")
    ax_pr.axhline(curves[0][3].mean(), color="0.5", lw=0.6, ls=":")  # no-skill
    ax_roc.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--")
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate",
               title="ROC curve")
    ax_pr.set(xlabel="recall", ylabel="precision", title="Precision-Recall")
    # Anchor the legends well below the x-axis labels so they never overlap.
    ax_roc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32),
                  ncol=2, frameon=False, fontsize=5.5)
    ax_pr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32),
                 ncol=2, frameon=False, fontsize=5.5)
    fig.suptitle("Brandon McKay population: tuned models, held-out test", y=1.02)
    fig.tight_layout()
    fig.savefig(ROC_OUT, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"\nwrote {ROC_OUT}")
    return 0


# CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Figures over the McKay population (analysis, "
                    "permutation importance, ROC/PR).")
    p.add_argument("--fig", choices=["all", "analysis", "importance", "roc-pr"],
                   default="all", help="which figure set to produce (default all)")
    # importance flags
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                   help="seeds for the importance sweep (default 0..4)")
    p.add_argument("--repeats", type=int, default=20,
                   help="permutation repeats per feature per seed (default 20)")
    # roc-pr flags
    p.add_argument("--seed", type=int, default=0,
                   help="split seed for the roc-pr figure (default 0)")
    args = p.parse_args(list(argv) if argv is not None else None)

    rc = 0
    if args.fig in ("all", "analysis"):
        rc = run_analysis(args) or rc
    if args.fig in ("all", "importance"):
        warnings.filterwarnings("ignore")
        rc = run_importance(args) or rc
    if args.fig in ("all", "roc-pr"):
        warnings.filterwarnings("ignore")
        rc = run_roc_pr(args) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
