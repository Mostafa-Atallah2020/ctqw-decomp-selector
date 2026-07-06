"""Logistic-regression decomposition selector.

Frames the task as binary classification: given a graph's topology features,
predict whether MATCHING wins (delta_cx > 0) or PAULI wins (delta_cx < 0). Ties
(delta_cx == 0) are excluded. Logistic regression is the only model for this
project: the goal is the win/lose decision, not the magnitude of the gap.

One L2-regularized logistic regression is fit PER CORPUS (er, structured,
hybrid) on the 10 standardized features, with a leakage-safe train/val/test
split stratified by (size, class): the StandardScaler is fit on the training
rows only. The same estimator both traces the training curve (warm_start, a
fixed number of solver iterations) and reports the final held-out test metrics,
so the curve and the reported numbers come from the identical fit.

(A size-extrapolation holdout, training on n < 128 and testing on n = 128, is
available behind RUN_EXTRAPOLATION but disabled by default: n = 128 is the
largest labeled size and its two classes are trivially separable, so it is a
weak check. True extrapolation to bigger n is future work, see HOLDOUT_N.)

Outputs (to results/model/):

  - metrics.csv           : held-out TEST metrics per corpus (accuracy,
                            precision/recall/specificity/NPV, F1, MCC, Cohen's
                            kappa, ROC-AUC, PR-AUC, log-loss, CX savings
                            captured, and confusion-matrix counts).
  - training_history.csv  : per-iteration train AND validation metrics per
                            corpus (the progress curves).
  - test_predictions.csv  : per-graph held-out test predictions (y_true, proba,
                            n_vertices), feeding the ROC/PR/confusion/calibration
                            plots.
  - coefficients.csv      : learned per-corpus feature weights.
  - feature_regressions.csv : per-corpus feature values + delta_cx for the
                            feature-vs-target and decision-boundary plots.

CX SAVINGS CAPTURED is the project's north-star: of all the CX gates matching
could save over Pauli on the test set, what fraction does the model's "use
matching" recommendation actually claim?

This module is the importable library; run it via scripts/train_model.py.
"""

from __future__ import annotations

import csv
import logging
import warnings
from pathlib import Path

# Reuse the analysis module's corpus loader, feature list, and target column so
# the model trains on exactly the columns the analysis reported on.
from utils.analysis.corpus_analysis import (
    DEFAULT_DATA,
    _model_features,
    load_corpus,
)

log = logging.getLogger("model")

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "model"

RANDOM_STATE = 0
HOLDOUT_N = 128          # size-extrapolation holdout target (see module docstring)
RUN_EXTRAPOLATION = False
N_ITERS = 50             # fixed lbfgs iterations traced for the training curve


# Corpora to train on, in plot order.
CORPORA = ["er", "structured", "hybrid"]


def _load_binary(data_dir: Path, corpus: str):
    """Load one corpus, drop ties, and build the binary target.

    Returns (df, feature_cols). `df` gains a `matching_wins` column in {0, 1}.
    """
    df = load_corpus(data_dir, corpus)
    df = df[df.delta_cx != 0].copy()
    df["matching_wins"] = (df.delta_cx > 0).astype(int)
    return df, _model_features()


def _savings_captured(delta_cx, y_pred) -> float:
    """CX savings captured: fraction of matching's total realizable CX savings
    that the model claims by recommending matching.

    numerator   = sum of true delta_cx over graphs the model sends to matching
                  (negative delta_cx among them are real losses a wrong "use
                  matching" call incurs).
    denominator = sum of true delta_cx over graphs where matching truly wins.
    """
    import numpy as np
    delta_cx = np.asarray(delta_cx, dtype=float)
    y_pred = np.asarray(y_pred)
    realizable = delta_cx[delta_cx > 0].sum()
    if realizable <= 0:
        return float("nan")
    claimed = delta_cx[y_pred == 1].sum()
    return float(claimed / realizable)


def _metrics(name, split, clf, Xte, yte, dcx_te) -> dict:
    """Score a fitted classifier on a test split. Returns a flat dict of the
    standard binary-classification metrics from the literature plus the
    project's CX savings captured."""
    import numpy as np
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

    y_pred = clf.predict(Xte)

    # Probability-based metrics need scores; guard degenerate cases.
    proba = None
    auc = ap = ll = float("nan")
    try:
        proba = clf.predict_proba(Xte)[:, 1]
        if len(np.unique(yte)) > 1 and len(np.unique(proba)) > 1:
            auc = float(roc_auc_score(yte, proba))
            ap = float(average_precision_score(yte, proba))
        ll = float(log_loss(yte, proba, labels=[0, 1]))
    except (AttributeError, ValueError):
        pass

    tn, fp, fn, tp = confusion_matrix(yte, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")   # negative predictive value

    def r(x):
        return round(float(x), 4) if x == x else ""   # blank on NaN

    return {
        "model": name,
        "split": split,
        "n": int(len(yte)),
        "accuracy": r(accuracy_score(yte, y_pred)),
        "precision": r(precision_score(yte, y_pred, zero_division=0)),      # PPV
        "recall": r(recall_score(yte, y_pred, zero_division=0)),            # sensitivity
        "specificity": r(specificity),
        "npv": r(npv),
        "f1": r(f1_score(yte, y_pred, zero_division=0)),
        "mcc": r(matthews_corrcoef(yte, y_pred)),
        "cohen_kappa": r(cohen_kappa_score(yte, y_pred)),
        "roc_auc": r(auc),
        "pr_auc": r(ap),
        "log_loss": r(ll),
        "savings_captured": r(_savings_captured(dcx_te, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _model():
    """The single model estimator used for BOTH the training curve and the final
    reported metrics. warm_start lets us trace its convergence one iteration at a
    time (see _train_history) without switching to a different estimator."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, warm_start=True,
                              random_state=RANDOM_STATE)


def _epoch_scores(prefix, y, proba, dcx, classes) -> dict:
    """The full metric set for one fold at one epoch, keyed with `prefix`
    ('train'/'val'). Same definitions as the final test table (_metrics), so the
    training curves and the results table are directly comparable."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, average_precision_score,
        cohen_kappa_score, confusion_matrix, f1_score, log_loss,
        matthews_corrcoef, precision_score, recall_score, roc_auc_score)

    pred = (proba >= 0.5).astype(int)
    auc = ap = float("nan")
    if len(np.unique(y)) > 1 and len(np.unique(proba)) > 1:
        auc = float(roc_auc_score(y, proba))
        ap = float(average_precision_score(y, proba))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    def r(x):
        return round(float(x), 4) if x == x else ""

    return {
        f"{prefix}_loss": round(float(log_loss(y, proba, labels=classes)), 5),
        f"{prefix}_accuracy": r(accuracy_score(y, pred)),
        f"{prefix}_precision": r(precision_score(y, pred, zero_division=0)),
        f"{prefix}_recall": r(recall_score(y, pred, zero_division=0)),
        f"{prefix}_specificity": r(specificity),
        f"{prefix}_npv": r(npv),
        f"{prefix}_f1": r(f1_score(y, pred, zero_division=0)),
        f"{prefix}_mcc": r(matthews_corrcoef(y, pred)),
        f"{prefix}_cohen_kappa": r(cohen_kappa_score(y, pred)),
        f"{prefix}_roc_auc": r(auc),
        f"{prefix}_pr_auc": r(ap),
        f"{prefix}_savings": r(_savings_captured(dcx, pred)),
    }


def _train_history(model, Xtr, ytr, dcx_tr, Xva, yva, dcx_va):
    """Trace `model`'s convergence, logging train AND validation metrics at each
    iteration, then return (history_rows, fitted_model) so the SAME estimator is
    used for the reported metrics.

    Uses scikit-learn's official callback API (SLEP023, sklearn >= 1.10): a
    custom FitCallback scores the full train/val metric set at each solver
    iteration via `on_fit_task_end`, reading the per-iteration model off the
    context. This replaces the earlier `warm_start`/`max_iter` refit trick (which
    was needed only because older sklearn exposed no per-iteration hook; see
    issue #26494, resolved by PR #33322/#33847).

    Fallback: if the installed sklearn lacks the callback API, fall back to the
    warm_start trace so the function still works on older versions.
    """
    import numpy as np
    from sklearn.base import clone
    classes = np.array([0, 1])

    def score_clf(iteration, clf):
        ptr = clf.predict_proba(Xtr)[:, 1]
        pva = clf.predict_proba(Xva)[:, 1]
        row = {"iteration": iteration}
        row.update(_epoch_scores("train", ytr, ptr, dcx_tr, classes))
        row.update(_epoch_scores("val", yva, pva, dcx_va, classes))
        return row

    # --- Official callback path (sklearn >= 1.10) ------------------------- #
    try:
        from sklearn.callback import FitCallback

        class _MetricTracer(FitCallback):
            """Records the full metric set at each lbfgs iteration."""
            def __init__(self):
                self.rows = []
            def on_fit_task_begin(self, estimator, context, **kw):
                pass
            def on_fit_task_end(self, estimator, context, *, fitted_estimator=None,
                                **kw):
                clf = fitted_estimator if fitted_estimator is not None else estimator
                # only the per-iteration subtasks carry a usable model
                if not hasattr(clf, "coef_"):
                    return
                self.rows.append(score_clf(len(self.rows) + 1, clf))
            def setup(self, estimator, context):
                pass
            def teardown(self, estimator, context):
                pass

        clf = clone(model)
        tracer = _MetricTracer()
        clf.set_callbacks(tracer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xtr, ytr)
        if tracer.rows:
            # renumber sequentially in case the root 'fit' task also fired
            for i, r in enumerate(tracer.rows, start=1):
                r["iteration"] = i
            return tracer.rows, clf
        # callback produced nothing usable -> fall through to warm_start
    except ImportError:
        pass

    # --- Fallback: warm_start trace (older sklearn) ----------------------- #
    history = []
    if getattr(model, "warm_start", False):
        clf = clone(model)
        clf.set_params(warm_start=True)
        for step in range(1, N_ITERS + 1):
            clf.set_params(max_iter=step)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(Xtr, ytr)
            history.append(score_clf(step, clf))
        return history, clf

    clf = clone(model).fit(Xtr, ytr)
    history.append(score_clf(1, clf))
    return history, clf


def _stratify_key(df):
    """Stratification key: (vertex count, class), so every size is split within
    itself and keeps its 50/50 class balance."""
    return df.n_vertices.astype(str) + "_" + df.matching_wins.astype(str)


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _split_three(df):
    """Train/val/test split (~64/16/20), stratified by (size, class) when the
    minority class is large enough; falls back to plain random otherwise (the
    ER corpus has only ~20 matching-wins, too few to stratify safely)."""
    from sklearn.model_selection import train_test_split
    strat = _stratify_key(df)
    # Every stratum needs >= 2 members to split; else stratify on the label only,
    # else no stratification at all.
    if strat.value_counts().min() >= 2:
        key1 = strat
    elif df.matching_wins.value_counts().min() >= 4:
        key1 = df.matching_wins
    else:
        key1 = None
    dev_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=key1, random_state=RANDOM_STATE)
    dev = df.loc[dev_idx]
    key2 = (df.loc[dev_idx].matching_wins
            if key1 is not None and dev.matching_wins.value_counts().min() >= 2
            else None)
    tr_idx, val_idx = train_test_split(
        dev.index, test_size=0.2, stratify=key2, random_state=RANDOM_STATE)
    return df.loc[tr_idx], df.loc[val_idx], df.loc[test_idx], dev


def _train_one_corpus(corpus, data_dir, feats):
    """Train logistic regression on one corpus. Returns a dict with the history
    rows, the test-metrics row, the held-out test predictions (for ROC/PR/
    confusion/calibration plots), and the learned coefficients."""
    from sklearn.preprocessing import StandardScaler

    df, _ = _load_binary(data_dir, corpus)
    tr, val, test, dev = _split_three(df)
    log.info("%s: %d graphs (%d matching / %d pauli); "
             "split %d train / %d val / %d test", corpus, len(df),
             int(df.matching_wins.sum()), int((df.matching_wins == 0).sum()),
             len(tr), len(val), len(test))

    scaler = StandardScaler().fit(tr[feats].to_numpy(float))
    def X(frame):
        return scaler.transform(frame[feats].to_numpy(float))

    # Trace the SAME estimator's convergence on train vs val (the curve)...
    history, _ = _train_history(
        _model(),
        X(tr), tr.matching_wins.to_numpy(), tr.delta_cx.to_numpy(float),
        X(val), val.matching_wins.to_numpy(), val.delta_cx.to_numpy(float))
    for row in history:                            # tag for grouping in plots
        row["corpus"] = corpus

    # ...then refit the same model class on train+val for the reported metrics.
    clf = _model().fit(X(dev), dev.matching_wins.to_numpy())
    y_true = test.matching_wins.to_numpy()
    test_row = _metrics("logistic", corpus, clf, X(test), y_true,
                        test.delta_cx.to_numpy(float))

    # Held-out test predictions (probability of "matching wins"), tagged with
    # the graph size, for the threshold/probability diagnostic plots (incl. the
    # per-n confusion matrices).
    proba = clf.predict_proba(X(test))[:, 1]
    nv = test.n_vertices.to_numpy()
    preds = [{"corpus": corpus, "n_vertices": int(n),
              "y_true": int(t), "proba": round(float(p), 6)}
             for n, t, p in zip(nv, y_true, proba)]

    # Learned coefficients on standardized features (the interpretability story).
    # Also save the scaler's mean/scale per feature so the exact trained decision
    # function can be reconstructed on RAW feature values downstream (e.g. the
    # decision-boundary plot projects this real model, rather than refitting one).
    coefs = [{"corpus": corpus, "feature": f, "coef": round(float(w), 6),
              "mean": round(float(m), 6), "scale": round(float(s), 6)}
             for f, w, m, s in zip(feats, clf.coef_.ravel(),
                                   scaler.mean_, scaler.scale_)]
    coefs.append({"corpus": corpus, "feature": "(intercept)",
                  "coef": round(float(clf.intercept_[0]), 6),
                  "mean": 0.0, "scale": 1.0})

    return {"history": history, "test_row": test_row,
            "preds": preds, "coefs": coefs}


def run(data_dir: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT) -> dict:
    """Train logistic regression separately on each corpus (er, structured,
    hybrid) with its own train/val/test split.

    Writes:
      - training_history.csv : per-epoch train AND validation metrics, per
                               corpus (the progress curves).
      - metrics.csv          : final held-out TEST metrics per corpus (the
                               results table), from each model refit on train+val.
    """
    feats = _model_features()
    all_history, all_test, all_preds, all_coefs = [], [], [], []
    for corpus in CORPORA:
        if not (data_dir / corpus / "labels" / "labels.csv").exists():
            log.warning("skipping '%s': no labels", corpus)
            continue
        r = _train_one_corpus(corpus, data_dir, feats)
        all_history.extend(r["history"])
        all_test.append(r["test_row"])
        all_preds.extend(r["preds"])
        all_coefs.extend(r["coefs"])

    # Put corpus/epoch first in the history CSV for readability.
    ordered = [{"corpus": r["corpus"], **{k: v for k, v in r.items()
                                          if k != "corpus"}}
               for r in all_history]
    _write_csv(out_dir / "training_history.csv", ordered)
    _write_csv(out_dir / "metrics.csv", all_test)
    _write_csv(out_dir / "test_predictions.csv", all_preds)
    _write_csv(out_dir / "coefficients.csv", all_coefs)

    # Feature values + signed cost gap per corpus, for the feature-vs-Δ_CX
    # regression plot (how each property drives matching-favorability).
    import pandas as pd
    reg_frames = []
    for corpus in CORPORA:
        if (data_dir / corpus / "labels" / "labels.csv").exists():
            d, _ = _load_binary(data_dir, corpus)
            sub = d[feats + ["delta_cx"]].copy()
            sub.insert(0, "corpus", corpus)
            reg_frames.append(sub)
    if reg_frames:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(reg_frames, ignore_index=True).round(6).to_csv(
            out_dir / "feature_regressions.csv", index=False)

    log.info("wrote metrics.csv, training_history.csv, test_predictions.csv, "
             "coefficients.csv, feature_regressions.csv to %s", out_dir)

    return {
        "n_features": len(feats),
        "features": feats,
        "target": "matching_wins = 1[delta_cx > 0]",
        "corpora": [r["split"] for r in all_test],  # split == corpus name here
        "test_metrics": all_test,
        "out_dir": str(out_dir),
    }
