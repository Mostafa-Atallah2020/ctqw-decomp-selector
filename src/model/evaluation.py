#!/usr/bin/env python3
"""Benchmark the shipped model against alternatives spanning different inductive
biases (linear, depth-limited trees, k-NN/RBF-SVM, ANN, tree ensembles).

Writes:
  results/evaluation/model_comparison.csv  every model x dataset, all metrics
  results/evaluation/seeds.csv             per-seed metrics for the shipped model
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

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from analysis.dataset_analysis import _model_features
from progress import Progress

FEATURES = _model_features()
DATASETS = ["er", "structured", "balanced"]
NMAX = 256             # largest labeled size
SEEDS = [0, 1, 2, 3, 4]
OUT = _ROOT / "results" / "evaluation"

METRICS = ["accuracy", "precision", "recall", "specificity", "npv", "f1", "mcc",
           "cohen_kappa", "roc_auc", "pr_auc", "log_loss", "savings"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load(dataset: str) -> pd.DataFrame:
    """Features inner-joined to labels, so only LABELED graphs are used. Ties are
    dropped. The target is y = 1[delta_cx > 0] ("matching is cheaper")."""
    d = _ROOT / "data" / dataset
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
    """The same metrics as results/logistic_model/metrics.csv, plus raw confusion
    counts. `score` (a probability) is needed for the AUCs and log-loss."""
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) > 1

    # CX savings: delta_cx summed over graphs sent to matching, over a perfect
    # chooser's realizable total. 1.0 is perfect, wrong choices push it below zero.
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
        # log-loss needs probabilities, an SVM decision function is not one
        m["log_loss"] = (log_loss(y_true, s, labels=[0, 1])
                         if s.min() >= 0 and s.max() <= 1 else float("nan"))
    else:
        m["roc_auc"] = m["pr_auc"] = m["log_loss"] = float("nan")
    return m


def score_model(clf, train, test, needs_scale: bool = True) -> dict:
    """Fit on train, score on test. The scaler is fit on training rows only.

    The unscaled path keeps DataFrames (not numpy) so fit and predict agree on
    feature names: LightGBM warns if fit with names but predicted on a bare array.
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
# the model comparison
# ---------------------------------------------------------------------------
def learned_models() -> dict:
    """{name: (estimator, needs_standardized_features)}, spanning different
    inductive biases. Trees are scale-invariant and take raw features."""
    m = {
        "logistic": (LogisticRegression(max_iter=1000, random_state=0), True),
        "naive_bayes": (GaussianNB(), True),
        # Swept at three depths so the effect of further splits is visible.
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
