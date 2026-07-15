#!/usr/bin/env python3
"""Leave-one-family-out: does the model learn which decomposition is cheaper, or
merely which generator produced the graph?

An ordinary train/test split cannot answer this, because both splits contain both
graph families. And the concern is real: our two families have near-opposite labels.
ER graphs are almost all Pauli-wins, structured graphs almost all matching-wins, so a
model could score highly just by recognising the generator's fingerprint ("this looks
like an ER graph, say Pauli"), without reasoning about decomposition cost at all.

Leave-one-family-out (LOFO) removes that crutch: train on one family, test on the
ENTIRE held-out family. The distributions are disjoint, so no fingerprint is
available at test time. Two folds, which is all there are:

    structured -> er       train on every structured graph, test on every ER graph
    er -> structured       and the reverse.

The hybrid corpus itself cannot appear here. It is sampled FROM er and structured, so
it owns no graphs of its own: every one of its 13,442 labeled graphs is already an ER
or a structured graph. Holding it out would mean testing on graphs that are in the
training set, and training on it would mean the held-out family is half of what the
model just trained on.

Writes results/evaluation/lofo.csv (one row per fold and seed, with the same twelve
metrics reported in results/logistic_model/metrics.csv).
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
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from utils.analysis.corpus_analysis import _model_features
from utils.progress import Progress

FEATURES = _model_features()

FOLDS = [
    ("structured", "er"),
    ("er", "structured"),
]
CORPORA = sorted({c for fold in FOLDS for c in fold})

NMAX = 256             # the largest labeled size
SEEDS = [0, 1, 2, 3, 4]
OUT = _ROOT / "results" / "evaluation"


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


def metrics(y_true, y_pred, dcx, score) -> dict:
    """The same twelve metrics reported in results/logistic_model/metrics.csv.

    Accuracy is not the headline by accident: on ER a constant predictor already
    scores 99.8%, so MCC and CX-savings-captured are what we read.
    """
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    corpora = {c: load(c) for c in CORPORA}

    prog = Progress(total=len(FOLDS) * len(SEEDS), unit="fit")
    print(f"fitting {prog.total} models "
          f"({len(FOLDS)} folds x {len(SEEDS)} seeds)")

    rows = []
    for train_on, test_on in FOLDS:
        train_df, test_df = corpora[train_on], corpora[test_on]

        for seed in SEEDS:
            # The scaler is fit on training rows only, so nothing leaks.
            scaler = StandardScaler().fit(train_df[FEATURES])
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(scaler.transform(train_df[FEATURES]), train_df.y)

            X_test = scaler.transform(test_df[FEATURES])
            row = metrics(test_df.y.to_numpy(), clf.predict(X_test),
                          test_df.delta_cx.to_numpy(),
                          clf.predict_proba(X_test)[:, 1])
            row.update({
                "train_on": train_on,
                "test_on": test_on,
                "seed": seed,
                "n_train": len(train_df),
                "n_test": len(test_df),
            })
            rows.append(row)
            prog.step(f"{train_on} -> {test_on} seed={seed}")
    prog.done()

    lofo = pd.DataFrame(rows)
    lofo.to_csv(OUT / "lofo.csv", index=False)
    print(f"wrote {OUT / 'lofo.csv'}")

    print(f"\nleave-one-family-out (mean over {len(SEEDS)} seeds):\n")
    print(f"  {'train -> test':<26}{'MCC':>8}{'recall':>9}{'savings':>12}")

    summary = (lofo.groupby(["train_on", "test_on"])
                   [["accuracy", "recall", "mcc", "savings"]].mean())
    for train_on, test_on in FOLDS:                 # keep the declared fold order
        r = summary.loc[(train_on, test_on)]
        print(f"  {train_on + ' -> ' + test_on:<26}{r.mcc:>8.3f}{r.recall:>9.3f}"
              f"{r.savings:>12.3f}")

    print(f"\n  Compare with MCC ~0.96 on the hybrid (results/logistic_model/metrics.csv), "
          "where the\n  model may use the family fingerprint. Falling to "
          f"{summary.mcc.min():.3f} means it does not\n  transfer: the features that "
          "separate the classes within a family partly encode\n  WHICH family the "
          "graph came from, so holding a family out strips away a cue\n  the model "
          "was leaning on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
