#!/usr/bin/env python3
"""Drop-one-feature ablation: which features does the model actually DEPEND on?

This is deliberately a different question from the univariate feature importance in
the exploratory analysis. That one scores each feature *in isolation* and is blind
to redundancy: two collinear features can each look important while being
interchangeable, so deleting either costs nothing. Our correlation heatmap shows
the degree-based features are strongly collinear, so that blind spot is live here.

The ablation measures dependence instead: retrain with one feature removed and see
how much the model degrades. A feature whose removal costs nothing was not
carrying unique signal, however informative it looked on its own.

Writes results/evaluation/ablation.csv: per dropped feature, the same twelve metrics
reported in results/logistic_model/metrics.csv (mean and std across seeds), plus each one's
drop relative to the full model. Rows are ranked by the MCC drop, but the savings
column matters too: a feature can cost little accuracy and still cost real CX gates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from utils.analysis.corpus_analysis import _model_features
from utils.progress import Progress

FEATURES = _model_features()
CORPUS = "hybrid"      # the only corpus balanced enough to train on
NMAX = 256             # the largest labeled size; bigger graphs exist but are unlabeled
SEEDS = [0, 1, 2, 3, 4]
OUT = _ROOT / "results" / "evaluation"

METRICS = ["accuracy", "precision", "recall", "specificity", "npv", "f1", "mcc",
           "cohen_kappa", "roc_auc", "pr_auc", "log_loss", "savings"]


def load(corpus: str) -> pd.DataFrame:
    """Features inner-joined to labels, so only LABELED graphs are used. Ties are
    dropped; the target is y = 1[delta_cx > 0], i.e. "matching is cheaper"."""
    d = _ROOT / "data" / corpus
    feat = pd.read_csv(d / "features" / "graph_features.csv")
    lab = pd.read_csv(d / "labels" / "labels.csv")

    df = feat.merge(lab[["graph_id", "delta_cx"]], on="graph_id", how="inner")
    df = df[(df.n_vertices <= NMAX) & (df.delta_cx != 0)].copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    return df


def split(df: pd.DataFrame, seed: int):
    """Leakage-safe ~64/16/20 split, stratified by (size, class) so every vertex
    count appears in every part: a model cannot win by learning "large => match"."""
    def key(frame):
        k = frame.n_vertices.astype(str) + "_" + frame.y.astype(str)
        return k if k.value_counts().min() >= 2 else None

    dev, test = train_test_split(df, test_size=0.2, random_state=seed,
                                 stratify=key(df))
    train, val = train_test_split(dev, test_size=0.2, random_state=seed,
                                  stratify=key(dev))
    return train, val, test


def metrics(y_true, y_pred, dcx, score) -> dict:
    """The same twelve metrics reported in results/logistic_model/metrics.csv."""
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) > 1

    # CX savings captured. delta_cx is CX_pauli - CX_matching, so a positive value
    # means matching is cheaper by that many gates.
    dcx = np.asarray(dcx, dtype=float)
    realizable = dcx[dcx > 0].sum()
    savings = (float(dcx[np.asarray(y_pred) == 1].sum() / realizable)
               if realizable > 0 else float("nan"))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if both else 0.0,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, score) if both else float("nan"),
        "pr_auc": average_precision_score(y_true, score) if both else float("nan"),
        "log_loss": log_loss(y_true, score, labels=[0, 1]) if both else float("nan"),
        "savings": savings,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def score_with(train, test, features: list[str]) -> dict:
    """Fit the shipped model on a feature subset; score it on the test split. The
    scaler is fit on training rows only, so nothing leaks."""
    scaler = StandardScaler().fit(train[features])
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(scaler.transform(train[features]), train.y)

    X_test = scaler.transform(test[features])
    return metrics(test.y.to_numpy(), clf.predict(X_test),
                   test.delta_cx.to_numpy(), clf.predict_proba(X_test)[:, 1])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load(CORPUS)

    # One refit per (seed, dropped-feature), plus one full model per seed.
    prog = Progress(total=len(SEEDS) * (len(FEATURES) + 1), unit="refit")
    print(f"refitting {prog.total} models "
          f"({len(FEATURES)} features + full, x {len(SEEDS)} seeds)")

    rows = []
    for seed in SEEDS:
        train, _val, test = split(df, seed)

        row = score_with(train, test, FEATURES)
        row.update({"dropped": "(none)", "seed": seed})
        rows.append(row)
        prog.step(f"seed={seed} full model")

        for feature in FEATURES:
            kept = [f for f in FEATURES if f != feature]
            row = score_with(train, test, kept)
            row.update({"dropped": feature, "seed": seed})
            rows.append(row)
            prog.step(f"seed={seed} without {feature}")
    prog.done()

    per_seed = pd.DataFrame(rows)
    ablation = (per_seed.groupby("dropped")[METRICS]
                .agg(["mean", "std"]))
    # Flatten the ("mcc", "mean") column MultiIndex to "mcc_mean".
    ablation.columns = [f"{m}_{stat}" for m, stat in ablation.columns]
    ablation = ablation.reset_index()

    # Rank by what the model is FOR: the drop in MCC when the feature is removed.
    full = ablation.loc[ablation.dropped == "(none)"].iloc[0]
    for metric in METRICS:
        ablation[f"{metric}_drop"] = full[f"{metric}_mean"] - ablation[f"{metric}_mean"]
    ablation = ablation.sort_values("mcc_drop", ascending=False)

    ablation.to_csv(OUT / "ablation.csv", index=False)
    print(f"wrote {OUT / 'ablation.csv'}")

    print(f"\ndrop-one-feature ablation on the {CORPUS} corpus "
          f"({len(df)} graphs, mean over {len(SEEDS)} seeds):\n")
    print(f"  {'dropped':<22}{'MCC':>8}{'drop':>9}{'savings':>10}{'drop':>9}")
    for _, r in ablation.iterrows():
        name = "full model" if r.dropped == "(none)" else f"without {r.dropped}"
        drop_mcc = "" if r.dropped == "(none)" else f"{r.mcc_drop:+9.4f}"
        drop_sav = "" if r.dropped == "(none)" else f"{r.savings_drop:+9.4f}"
        print(f"  {name:<22}{r.mcc_mean:>8.4f}{drop_mcc:>9}"
              f"{r.savings_mean:>10.4f}{drop_sav:>9}")

    print("\nA near-zero drop means the feature carried no UNIQUE signal: the "
          "others already encode it.\nAll twelve metrics are in ablation.csv; MCC "
          "and CX savings are shown here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
