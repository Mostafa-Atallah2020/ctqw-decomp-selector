#!/usr/bin/env python3
"""Predict on the UNLABELED graphs in the corpus, and score those predictions
with the label-free estimators from the literature (CLI entry point).

An unlabeled graph is one that has features in data/<corpus>/features/ but no row
in data/<corpus>/labels/, i.e. it was generated and featurized, but labeling it
was infeasible. Labeling cost grows as ~n^3.2 (measured), so it stops being
practical past n=256: ~0.5 h/graph at n=512 and ~4.7 h/graph at n=1024, dominated
by BUILDING the Pauli circuit rather than transpiling it.

This script does NOT generate graphs, and it does not assemble corpora. It is a pure
consumer of data/: every corpus it reports on, the hybrid included, is READ from disk,
so every prediction is traceable to a stored graph. Build them first:

    python scripts/generate_graphs.py  --corpus er         --vertices 512 1024 -n 100
    python scripts/generate_graphs.py  --corpus structured --vertices 512 1024 -n 100
    python scripts/extract_features.py --corpus er
    python scripts/extract_features.py --corpus structured
    python scripts/build_hybrid.py                 # carries the unlabeled graphs over
    python scripts/predict_unlabeled.py

WHAT THIS IS: a deployment demonstration plus a reliability assessment.
WHAT THIS IS NOT: a validation. With no labels at these sizes we cannot compute
accuracy, MCC, or CX savings. Everything reported is either a model output or a
label-free ESTIMATE whose own assumptions are strained under this much shift.

Label-free scores (each with its caveat):

  disagreement     Jiang et al., ICLR 2022, "Assessing Generalization of SGD via
                   Disagreement": the rate at which independently trained models
                   disagree on unlabeled data tracks the test error. ASYMMETRIC:
                   a high rate PROVES unreliability, while unanimity is only weak
                   evidence of correctness.
  mean_confidence  Average max-probability (Guillory et al. 2021). Models get
                   MORE confident and LESS accurate under shift (Ovadia et al.,
                   NeurIPS 2019), so this is an upper bound at best.
  atc_estimate     Average Thresholded Confidence (Garg et al., ICLR 2022): a
                   confidence threshold calibrated on the labeled validation
                   split; the fraction of unlabeled points above it estimates
                   accuracy. Assumes MILD shift.
  mahalanobis      Distance from the training feature distribution: how far out
                   of distribution these graphs actually are.
  features_oob     How many of the ten features fall outside the min/max range
                   ever seen in training (a crude but readable shift signal).

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

from utils.analysis.corpus_analysis import DEFAULT_DATA, _model_features
from utils.progress import Progress

FEATURES = _model_features()
CORPORA = ("er", "structured", "hybrid")
SEED = 42
OUT = _ROOT / "results" / "predictions"


def models() -> dict:
    """A deliberately diverse ensemble: the disagreement estimate is only
    meaningful if the models have genuinely different inductive biases. All of
    these score within noise of each other on the labeled test set (see the model
    ladder in benchmark_models.py), so any disagreement out here is about
    extrapolation, not about one model simply being worse."""
    return {
        "logistic": LogisticRegression(max_iter=1000, random_state=0),
        "dtree_d2": DecisionTreeClassifier(max_depth=2, random_state=0),
        "rand_forest": RandomForestClassifier(n_estimators=200, random_state=0),
        "grad_boost": GradientBoostingClassifier(random_state=0),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=0),
    }


# ---------------------------------------------------------------------------
# data: read what the pipeline already saved under data/
# ---------------------------------------------------------------------------
def _corpus_frames(corpus: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = DEFAULT_DATA / corpus
    feat = pd.read_csv(d / "features" / "graph_features.csv")
    lab = pd.read_csv(d / "labels" / "labels.csv")
    return feat, lab


def load_labeled() -> pd.DataFrame:
    """The labeled hybrid corpus: what the ensemble trains on."""
    feat, lab = _corpus_frames("hybrid")
    df = feat.merge(lab[["graph_id", "delta_cx"]], on="graph_id", how="inner")
    df = df[df.delta_cx != 0].copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    return df


def load_unlabeled(corpus: str) -> pd.DataFrame:
    """Graphs that have features but NO label: an anti-join against labels.csv.

    These are exactly the graphs the pipeline generated and featurized but could
    not afford to label. Nothing is generated here; if this comes back empty, the
    graphs have not been built yet (see the header for the two commands).
    """
    feat, lab = _corpus_frames(corpus)
    unlabeled = feat[~feat.graph_id.isin(set(lab.graph_id))].copy()
    unlabeled["corpus"] = corpus
    return unlabeled


# ---------------------------------------------------------------------------
def train_ensemble(labeled: pd.DataFrame):
    """Fit every model on the labeled hybrid; hold out a validation split so ATC's
    threshold can be calibrated on labeled data, as that method requires."""
    train, val = train_test_split(
        labeled, test_size=0.2, random_state=SEED,
        stratify=labeled.n_vertices.astype(str) + "_" + labeled.y.astype(str))

    scaler = StandardScaler().fit(train[FEATURES])
    X_train, X_val = scaler.transform(train[FEATURES]), scaler.transform(val[FEATURES])

    fitted = {name: clone(m).fit(X_train, train.y) for name, m in models().items()}
    print(f"ensemble trained on {len(train)} labeled hybrid graphs "
          f"(+{len(val)} held out to calibrate ATC); "
          f"sizes {sorted(labeled.n_vertices.unique())}")

    # ATC threshold: the confidence quantile at which the fraction of validation
    # points above it equals the model's validation accuracy.
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
    out = unlabeled[["graph_id", "corpus", "n_vertices"]].copy()

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
    """The label-free estimates, per (corpus, size)."""
    pred_cols = [f"pred_{n}" for n in names]
    rows = []
    for (corpus, size), g in predictions.groupby(["corpus", "n_vertices"]):
        preds = g[pred_cols].to_numpy()
        confidences, atc = {}, {}
        for name in names:
            p = g[f"proba_{name}"].to_numpy()
            c = np.maximum(p, 1 - p)
            confidences[name] = float(c.mean())
            atc[name] = float((c >= atc_threshold[name]).mean())
        rows.append({
            "corpus": corpus, "n_vertices": size, "n_graphs": len(g),
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
    p.add_argument("--corpus", nargs="+", default=list(CORPORA),
                   choices=list(CORPORA),
                   help="corpora to report on (default: all)")
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="restrict to these vertex counts (default: every "
                        "unlabeled size found in data/)")
    p.add_argument("--out", type=Path, default=OUT,
                   help="output dir (default: results/predictions)")
    args = p.parse_args(list(argv) if argv is not None else None)
    args.out.mkdir(parents=True, exist_ok=True)

    # The unlabeled graphs, straight from data/. 
    per_corpus = {c: load_unlabeled(c) for c in args.corpus}
    if args.sizes:
        per_corpus = {c: df[df.n_vertices.isin(args.sizes)]
                      for c, df in per_corpus.items()}

    for corpus, df in per_corpus.items():
        sizes = sorted(df.n_vertices.unique())
        print(f"{corpus:11s} {len(df):5d} unlabeled graphs in data/  sizes={sizes}")

    if not any(len(df) for df in per_corpus.values()):
        print("\nNo unlabeled graphs found. Generate and featurize them first:\n"
              "  python scripts/generate_graphs.py  --corpus er         "
              "--vertices 512 1024 -n 100\n"
              "  python scripts/generate_graphs.py  --corpus structured "
              "--vertices 512 1024 -n 100\n"
              "  python scripts/extract_features.py --corpus er\n"
              "  python scripts/extract_features.py --corpus structured\n"
              "  python scripts/build_hybrid.py")
        return 1

    unlabeled = pd.concat(per_corpus.values(), ignore_index=True)

    # Fit the ensemble on the labeled corpus, then predict.
    prog = Progress(total=1 + len(unlabeled.groupby(["corpus", "n_vertices"])),
                    unit="step")
    fitted, scaler, train, atc_threshold, shift = train_ensemble(load_labeled())
    prog.step(f"trained {len(models())} models on the labeled hybrid")

    parts = []
    for (corpus, size), group in unlabeled.groupby(["corpus", "n_vertices"]):
        parts.append(predict(group, fitted, scaler, train, shift))
        prog.step(f"predicted {corpus} n={size} ({len(group)} graphs)")
    prog.done()
    predictions = pd.concat(parts, ignore_index=True)
    names = list(fitted)
    scores = score(predictions, names, atc_threshold)

    # Per-model breakdown: the aggregate hides WHICH model dissents.
    per_model = []
    for (corpus, size), g in predictions.groupby(["corpus", "n_vertices"]):
        row = {"corpus": corpus, "n_vertices": size, "n_graphs": len(g)}
        for name in names:
            row[name] = round(100 * float(g[f"pred_{name}"].mean()), 1)
        row["disagreement"] = float(scores[
            (scores.corpus == corpus) & (scores.n_vertices == size)
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
          "PROVES the extrapolation is unreliable; unanimity is only weak evidence "
          "of correctness. On the labeled corpus, where we can check, disagreement "
          "is near zero and the true test MCC is ~0.96.")

    print(f"\n-> {args.out / 'unlabeled_predictions.csv'}")
    print(f"-> {args.out / 'per_model_predictions.csv'}")
    print(f"-> {args.out / 'label_free_scores.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
