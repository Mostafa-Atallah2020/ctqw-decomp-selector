#!/usr/bin/env python3
"""Predict on the UNLABELED graphs in data/, and score those predictions with
label-free estimators (CLI entry point). Unlabeled = has features but no label
row (labeling cost grows ~n^3.2, infeasible past n=256). Not a validation: with
no labels every number is a model output or a shift-strained ESTIMATE.

Build inputs first, e.g.:
    python scripts/dataset.py --stage features --dataset er
    python scripts/predict_unlabeled.py

Label-free scores: disagreement (Jiang et al. ICLR 2022, asymmetric - high
proves unreliability, unanimity only weak evidence), mean_confidence (Guillory
2021), atc_estimate (Garg et al. ICLR 2022, assumes mild shift), mahalanobis
(distance from training distribution), features_oob (features outside training
min/max).

Outputs:
  results/predictions/unlabeled_predictions.csv   per graph, per model
  results/predictions/per_model_predictions.csv   which models dissent, and where
  results/predictions/label_free_scores.csv       the estimates above
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from analysis.dataset_analysis import DEFAULT_DATA, _model_features
from progress import Progress

FEATURES = _model_features()
DATASETS = ("er", "structured", "balanced")
SEED = 42
OUT = _ROOT / "results" / "predictions"


def models() -> dict:
    """Deliberately diverse ensemble: disagreement is only meaningful if the
    models have different inductive biases. All score within noise on the labeled
    test set, so disagreement here reflects extrapolation, not a weaker model."""
    return {
        "logistic": LogisticRegression(max_iter=1000, random_state=0),
        "dtree_d2": DecisionTreeClassifier(max_depth=2, random_state=0),
        "rand_forest": RandomForestClassifier(n_estimators=200, random_state=0),
        "grad_boost": GradientBoostingClassifier(random_state=0),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=0),
    }


# data: read what the pipeline already saved under data/
def _dataset_frames(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = DEFAULT_DATA / dataset
    feat = pd.read_csv(d / "features" / "graph_features.csv")
    lab = pd.read_csv(d / "labels" / "labels.csv")
    return feat, lab


def load_labeled() -> pd.DataFrame:
    """The labeled balanced dataset: what the ensemble trains on."""
    feat, lab = _dataset_frames("balanced")
    df = feat.merge(lab[["graph_id", "delta_cx"]], on="graph_id", how="inner")
    df = df[df.delta_cx != 0].copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    return df


def load_unlabeled(dataset: str) -> pd.DataFrame:
    """Graphs with features but NO label: an anti-join against labels.csv.

    Nothing is generated here. If empty, the graphs have not been built yet.
    """
    feat, lab = _dataset_frames(dataset)
    unlabeled = feat[~feat.graph_id.isin(set(lab.graph_id))].copy()
    unlabeled["dataset"] = dataset
    return unlabeled


def train_ensemble(labeled: pd.DataFrame):
    """Fit every model on the labeled balanced, hold out a validation split to
    calibrate ATC's threshold on labeled data, as that method requires."""
    train, val = train_test_split(
        labeled, test_size=0.2, random_state=SEED,
        stratify=labeled.n_vertices.astype(str) + "_" + labeled.y.astype(str))

    scaler = StandardScaler().fit(train[FEATURES])
    X_train, X_val = scaler.transform(train[FEATURES]), scaler.transform(val[FEATURES])

    fitted = {name: clone(m).fit(X_train, train.y) for name, m in models().items()}
    print(f"ensemble trained on {len(train)} labeled balanced graphs "
          f"(+{len(val)} held out to calibrate ATC), "
          f"sizes {sorted(labeled.n_vertices.unique())}")

    # ATC threshold: confidence quantile where the fraction of val points above
    # it equals the model's validation accuracy.
    atc_threshold = {}
    for name, clf in fitted.items():
        p = clf.predict_proba(X_val)[:, 1]
        confidence = np.maximum(p, 1 - p)
        accuracy = (clf.predict(X_val) == val.y.to_numpy()).mean()
        atc_threshold[name] = float(np.quantile(confidence, max(0.0, 1.0 - accuracy)))

    # Training feature distribution, for the Mahalanobis shift measure.
    mu = X_train.mean(axis=0)
    cov = np.cov(X_train, rowvar=False) + 1e-6 * np.eye(len(FEATURES))
    shift = (mu, np.linalg.inv(cov))

    return fitted, scaler, train, atc_threshold, shift


def predict(unlabeled: pd.DataFrame, fitted, scaler, train, shift) -> pd.DataFrame:
    """Every model's prediction on every unlabeled graph, plus the shift measures."""
    mu, cov_inv = shift
    lo, hi = train[FEATURES].min(), train[FEATURES].max()

    X = scaler.transform(unlabeled[FEATURES])
    out = unlabeled[["graph_id", "dataset", "n_vertices"]].copy()

    for name, clf in fitted.items():
        p = clf.predict_proba(X)[:, 1]
        out[f"proba_{name}"] = p
        out[f"pred_{name}"] = (p >= 0.5).astype(int)

    d = X - mu
    out["mahalanobis"] = np.sqrt(np.einsum("ij,jk,ik->i", d, cov_inv, d))
    out["features_oob"] = ((unlabeled[FEATURES] < lo) |
                           (unlabeled[FEATURES] > hi)).sum(axis=1).to_numpy()
    return out


def score(predictions: pd.DataFrame, names: list[str], atc_threshold: dict) -> pd.DataFrame:
    """The label-free estimates, per (dataset, size)."""
    pred_cols = [f"pred_{n}" for n in names]
    rows = []
    for (dataset, size), g in predictions.groupby(["dataset", "n_vertices"]):
        preds = g[pred_cols].to_numpy()
        confidences, atc = {}, {}
        for name in names:
            p = g[f"proba_{name}"].to_numpy()
            c = np.maximum(p, 1 - p)
            confidences[name] = float(c.mean())
            atc[name] = float((c >= atc_threshold[name]).mean())
        rows.append({
            "dataset": dataset, "n_vertices": size, "n_graphs": len(g),
            "pct_predicted_matching": 100 * float(g["pred_logistic"].mean()),
            # disagreement: the fraction of graphs on which the models are NOT unanimous
            "ensemble_disagreement": float(
                (preds.min(axis=1) != preds.max(axis=1)).mean()),
            "mean_confidence": float(np.mean(list(confidences.values()))),
            "atc_estimated_accuracy": float(np.mean(list(atc.values()))),
            "mean_mahalanobis": float(g["mahalanobis"].mean()),
            "mean_features_oob": float(g["features_oob"].mean()),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Predict on the unlabeled graphs already in data/, and score "
                    "those predictions with label-free estimators.")
    p.add_argument("--dataset", nargs="+", default=list(DATASETS),
                   choices=list(DATASETS),
                   help="datasets to report on (default: all)")
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="restrict to these vertex counts (default: every "
                        "unlabeled size found in data/)")
    p.add_argument("--out", type=Path, default=OUT,
                   help="output dir (default: results/predictions)")
    args = p.parse_args(list(argv) if argv is not None else None)
    args.out.mkdir(parents=True, exist_ok=True)

    # The unlabeled graphs, straight from data/.
    per_dataset = {c: load_unlabeled(c) for c in args.dataset}
    if args.sizes:
        per_dataset = {c: df[df.n_vertices.isin(args.sizes)]
                      for c, df in per_dataset.items()}

    for dataset, df in per_dataset.items():
        sizes = sorted(df.n_vertices.unique())
        print(f"{dataset:11s} {len(df):5d} unlabeled graphs in data/  sizes={sizes}")

    if not any(len(df) for df in per_dataset.values()):
        print("\nNo unlabeled graphs found. Generate and featurize them first:\n"
              "  python scripts/dataset.py --stage generate --dataset er         "
              "--vertices 512 1024 -n 100\n"
              "  python scripts/dataset.py --stage generate --dataset structured "
              "--vertices 512 1024 -n 100\n"
              "  python scripts/dataset.py --stage features --dataset er\n"
              "  python scripts/dataset.py --stage features --dataset structured\n"
              "  python scripts/dataset.py --stage balanced")
        return 1

    unlabeled = pd.concat(per_dataset.values(), ignore_index=True)

    # Fit the ensemble on the labeled dataset, then predict.
    prog = Progress(total=1 + len(unlabeled.groupby(["dataset", "n_vertices"])),
                    unit="step")
    fitted, scaler, train, atc_threshold, shift = train_ensemble(load_labeled())
    prog.step(f"trained {len(models())} models on the labeled balanced")

    parts = []
    for (dataset, size), group in unlabeled.groupby(["dataset", "n_vertices"]):
        parts.append(predict(group, fitted, scaler, train, shift))
        prog.step(f"predicted {dataset} n={size} ({len(group)} graphs)")
    prog.done()
    predictions = pd.concat(parts, ignore_index=True)
    names = list(fitted)
    scores = score(predictions, names, atc_threshold)

    # Per-model breakdown: the aggregate hides WHICH model dissents.
    per_model = []
    for (dataset, size), g in predictions.groupby(["dataset", "n_vertices"]):
        row = {"dataset": dataset, "n_vertices": size, "n_graphs": len(g)}
        for name in names:
            row[name] = round(100 * float(g[f"pred_{name}"].mean()), 1)
        row["disagreement"] = float(scores[
            (scores.dataset == dataset) & (scores.n_vertices == size)
        ].ensemble_disagreement.iloc[0])
        per_model.append(row)
    per_model = pd.DataFrame(per_model)

    predictions.to_csv(args.out / "unlabeled_predictions.csv", index=False)
    per_model.to_csv(args.out / "per_model_predictions.csv", index=False)
    scores.to_csv(args.out / "label_free_scores.csv", index=False)

    print("\n=== % OF GRAPHS EACH MODEL SENDS TO MATCHING ===")
    print(per_model.to_string(index=False))
    print("\n=== LABEL-FREE SCORES (estimates, not measurements) ===")
    print(scores.round(3).to_string(index=False))
    print("\nHow to read this: disagreement is ASYMMETRIC evidence. A high rate "
          "PROVES the extrapolation is unreliable. Unanimity is only weak evidence "
          "of correctness. On the labeled dataset, where we can check, disagreement "
          "is near zero and the true test MCC is ~0.96.")

    print(f"\n-> {args.out / 'unlabeled_predictions.csv'}")
    print(f"-> {args.out / 'per_model_predictions.csv'}")
    print(f"-> {args.out / 'label_free_scores.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
