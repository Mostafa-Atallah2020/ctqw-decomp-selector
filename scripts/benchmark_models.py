#!/usr/bin/env python3
"""Benchmark the shipped model against every serious alternative.

The question: is the linear model we ship leaving accuracy on the table?

Twelve learned models, spanning deliberately different inductive biases so that a
better boundary shape for this problem, if one exists, has a fair chance to show up:

    linear          logistic, naive Bayes
    trees           decision trees at depth 1, 2, 3
    local/kernel    k-NN, RBF-SVM
    neural          MLP
    tree ensembles  random forest, gradient boosting, XGBoost, LightGBM

Writes:
  results/evaluation/model_comparison.csv  every model x corpus, all twelve metrics
  results/evaluation/seeds.csv             per-seed metrics for the shipped model,
                                           for confidence intervals on the headline
                                           numbers

"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from utils.analysis.corpus_analysis import _model_features
from utils.progress import Progress

FEATURES = _model_features()
CORPORA = ["er", "structured", "hybrid"]
NMAX = 256             # the largest labeled size
SEEDS = [0, 1, 2, 3, 4]
OUT = _ROOT / "results" / "evaluation"

METRICS = ["accuracy", "precision", "recall", "specificity", "npv", "f1", "mcc",
           "cohen_kappa", "roc_auc", "pr_auc", "log_loss", "savings"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load(corpus: str) -> pd.DataFrame:
    """Features inner-joined to labels, so only LABELED graphs are used (data/ also
    holds larger, unlabeled graphs; see scripts/predict_unlabeled.py). Ties are
    dropped; the target is y = 1[delta_cx > 0], i.e. "matching is cheaper"."""
    d = _ROOT / "data" / corpus
    feat = pd.read_csv(d / "features" / "graph_features.csv")
    lab = pd.read_csv(d / "labels" / "labels.csv")

    df = feat.merge(lab[["graph_id", "delta_cx"]], on="graph_id", how="inner")
    df = df[(df.n_vertices <= NMAX) & (df.delta_cx != 0)].copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    return df


def split(df: pd.DataFrame, seed: int):
    """Leakage-safe ~64/16/20 split, stratified by (size, class) so every test set
    spans every vertex count: a model cannot win by learning "large => match"."""
    def key(frame):
        k = frame.n_vertices.astype(str) + "_" + frame.y.astype(str)
        return k if k.value_counts().min() >= 2 else None

    dev, test = train_test_split(df, test_size=0.2, random_state=seed,
                                 stratify=key(df))
    train, val = train_test_split(dev, test_size=0.2, random_state=seed,
                                  stratify=key(dev))
    return train, val, test


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred, dcx, score=None) -> dict:
    """The same twelve metrics reported in results/logistic_model/metrics.csv, plus the raw
    confusion counts. `score` (a probability) is needed for the AUCs and log-loss."""
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) > 1

    # CX savings captured. delta_cx is CX_pauli - CX_matching, so a positive value
    # means matching is cheaper by that many gates.
    #
    #   denominator: every gate matching COULD save, i.e. a perfect chooser's total.
    #   numerator:   the gates the model's choices actually save. Summing delta_cx
    #                over the graphs it sent to matching means a correct choice adds
    #                a positive gap, and a wrong one (matching on a Pauli-favourable
    #                graph) adds a negative one.
    #
    # So 1.0 is perfect, and enough wrong choices push it below zero: the model then
    # spends more gates than it saves.
    dcx = np.asarray(dcx, dtype=float)
    realizable = dcx[dcx > 0].sum()
    savings = (float(dcx[np.asarray(y_pred) == 1].sum() / realizable)
               if realizable > 0 else float("nan"))

    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if both else 0.0,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "savings": savings,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if score is not None and both:
        s = np.asarray(score, dtype=float)
        m["roc_auc"] = roc_auc_score(y_true, s)
        m["pr_auc"] = average_precision_score(y_true, s)
        # log-loss needs probabilities; an SVM decision function is not one.
        m["log_loss"] = (log_loss(y_true, s, labels=[0, 1])
                         if s.min() >= 0 and s.max() <= 1 else float("nan"))
    else:
        m["roc_auc"] = m["pr_auc"] = m["log_loss"] = float("nan")
    return m


def score_model(clf, train, test, needs_scale: bool = True) -> dict:
    """Fit on train, score on test. The scaler is fit on training rows only.

    The unscaled path keeps DataFrames rather than dropping to numpy: LightGBM
    records the feature names it was fit with and warns when asked to predict on
    a bare array, so fit and predict must agree on whether names exist. Scaling
    returns a numpy array either way, which is consistent by construction.
    """
    if needs_scale:
        scaler = StandardScaler().fit(train[FEATURES])
        X_train = scaler.transform(train[FEATURES])
        X_test = scaler.transform(test[FEATURES])
    else:
        X_train, X_test = train[FEATURES], test[FEATURES]

    clf.fit(X_train, train.y)
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        s = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        s = clf.decision_function(X_test)
    else:
        s = None
    return metrics(test.y.to_numpy(), y_pred, test.delta_cx.to_numpy(), s)


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
def learned_models() -> dict:
    """{name: (estimator, needs_standardized_features)}, spanning deliberately
    different inductive biases: linear, depth-limited trees, local, kernel, neural,
    and tree ensembles. Trees are scale-invariant and take raw features."""
    m = {
        "logistic": (LogisticRegression(max_iter=1000, random_state=0), True),
        "naive_bayes": (GaussianNB(), True),
        # Swept at three depths rather than fixed at one, so the effect of allowing
        # further splits is visible in the table.
        "dtree_d1": (DecisionTreeClassifier(max_depth=1, random_state=0), False),
        "dtree_d2": (DecisionTreeClassifier(max_depth=2, random_state=0), False),
        "dtree_d3": (DecisionTreeClassifier(max_depth=3, random_state=0), False),
        "svm_rbf": (SVC(kernel="rbf", probability=True, random_state=0), True),
        "knn": (KNeighborsClassifier(n_neighbors=5), True),
        "mlp": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                              random_state=0), True),
        "rand_forest": (RandomForestClassifier(n_estimators=200, random_state=0),
                        False),
        "grad_boost": (GradientBoostingClassifier(random_state=0), False),
        "xgboost": (XGBClassifier(n_estimators=200, eval_metric="logloss",
                                  random_state=0, verbosity=0), False),
        "lightgbm": (LGBMClassifier(n_estimators=200, random_state=0, verbose=-1),
                     False),
    }
    return m


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Benchmark the twelve-model ladder on one or more corpora.")
    p.add_argument("--corpus", nargs="+", default=CORPORA,
                   help=f"corpora to evaluate (default: {' '.join(CORPORA)}). "
                        "census8 is the exhaustive set of all 11,117 connected "
                        "8-vertex graphs.")
    args = p.parse_args(list(argv) if argv is not None else None)
    corpora = args.corpus
    # Default run writes model_comparison.csv; a custom --corpus set writes a
    # suffixed file so it never clobbers the canonical three-corpus comparison.
    is_default = corpora == CORPORA
    out_name = ("model_comparison.csv" if is_default
                else f"model_comparison_{'_'.join(corpora)}.csv")

    OUT.mkdir(parents=True, exist_ok=True)
    n_models = len(learned_models())
    prog = Progress(total=len(corpora) * len(SEEDS) * n_models, unit="fit")
    print(f"fitting {prog.total} models "
          f"({n_models} models x {len(corpora)} corpora x {len(SEEDS)} seeds)")

    rows = []
    for corpus in corpora:
        df = load(corpus)
        per_model: dict[str, list[dict]] = {}

        for seed in SEEDS:
            train, _val, test = split(df, seed)
            for name, (estimator, needs_scale) in learned_models().items():
                per_model.setdefault(name, []).append(
                    score_model(clone(estimator), train, test, needs_scale))
                prog.step(f"{corpus}/{name} seed={seed}")

        for name, runs in per_model.items():
            agg = {"corpus": corpus, "model": name}
            for metric in METRICS:
                vals = np.array([r[metric] for r in runs], dtype=float)
                # A metric can be NaN across every seed by design: an SVM decision
                # function is not a probability, so log-loss is undefined for it.
                # Report NaN rather than let numpy warn about an empty slice.
                if np.isnan(vals).all():
                    agg[f"{metric}_mean"] = float("nan")
                    agg[f"{metric}_std"] = float("nan")
                else:
                    agg[f"{metric}_mean"] = np.nanmean(vals)
                    agg[f"{metric}_std"] = np.nanstd(vals)
            rows.append(agg)

    prog.done()

    ladder = pd.DataFrame(rows)
    ladder.to_csv(OUT / out_name, index=False)
    print(f"\nwrote {OUT / out_name}")

    # The shipped-model confidence-interval file is specific to the hybrid corpus, so
    # only produce it on the default run.
    if is_default:
        df = load("hybrid")
        seed_rows = []
        for seed in SEEDS:
            train, _val, test = split(df, seed)
            row = score_model(LogisticRegression(max_iter=1000, random_state=0),
                              train, test)
            row["seed"] = seed
            seed_rows.append(row)
        seeds = pd.DataFrame(seed_rows)
        seeds.to_csv(OUT / "seeds.csv", index=False)
        print(f"wrote {OUT / 'seeds.csv'}")
        print(f"\nshipped model across seeds: MCC {seeds.mcc.mean():.3f} "
              f"+/- {seeds.mcc.std():.3f}")

    # Print the ladder for each corpus, sorted by MCC, with savings alongside so the
    # "good MCC, negative savings" pattern is visible.
    for corpus in corpora:
        sub = ladder[ladder.corpus == corpus].sort_values("mcc_mean",
                                                           ascending=False)
        print(f"\n{corpus} ladder (MCC, mean +/- std over {len(SEEDS)} seeds):")
        for _, r in sub.iterrows():
            print(f"  {r.model:<17} MCC {r.mcc_mean:6.3f} +/- {r.mcc_std:.3f}"
                  f"   savings {r.savings_mean:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
