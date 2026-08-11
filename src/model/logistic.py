"""Logistic-regression decomposition selector.

Binary classification: from a graph's topology features, predict whether MATCHING
wins (delta_cx > 0) or PAULI wins (delta_cx < 0). Ties are excluded. One
L2-regularized fit PER DATASET (er, structured, balanced) on the 10 standardized
features, with a leakage-safe (size, class)-stratified train/val/test split
(StandardScaler fit on train rows only). The same estimator traces the training
curve and reports the final held-out test metrics.

Outputs (to results/logistic_model/): metrics.csv, training_history.csv,
test_predictions.csv, coefficients.csv, feature_regressions.csv.

Run via scripts/train_logistic.py.
"""

from __future__ import annotations

import csv
import logging
import warnings
from pathlib import Path

# Reuse the analysis module's loader/feature list so the model trains on exactly
# the columns the analysis reported on.
from analysis.dataset_analysis import (
    DEFAULT_DATA,
    _model_features,
    load_dataset,
)

log = logging.getLogger("model")

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO_ROOT / "results" / "logistic_model"

RANDOM_STATE = 0
HOLDOUT_N = 128          # size-extrapolation holdout target (see module docstring)
RUN_EXTRAPOLATION = False
N_ITERS = 50             # fixed lbfgs iterations traced for the training curve


# Datasets to train on, in plot order.
DATASETS = ["er", "structured", "balanced"]


def _load_binary(data_dir: Path, dataset: str):
    """Load one dataset, drop ties, add a `matching_wins` {0,1} column.

    Returns (df, feature_cols).
    """
    df = load_dataset(data_dir, dataset)
    df = df[df.delta_cx != 0].copy()
    df["matching_wins"] = (df.delta_cx > 0).astype(int)
    return df, _model_features()


def _savings_captured(delta_cx, y_pred) -> float:
    """CX savings captured: fraction of matching's total realizable CX savings
    the model claims by recommending matching.

    numerator = sum of true delta_cx over graphs sent to matching (wrong calls
    subtract their real losses). Denominator = sum over graphs matching truly wins.
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
    """Score a fitted classifier on a test split: flat dict of the standard
    binary-classification metrics plus CX savings captured."""
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

    # Probability-based metrics need scores, guard degenerate cases.
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
    """The single estimator used for BOTH the training curve and the final
    reported metrics. warm_start allows the per-iteration trace (see
    _train_history)."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, warm_start=True,
                              random_state=RANDOM_STATE)


def _epoch_scores(prefix, y, proba, dcx, classes) -> dict:
    """Full metric set for one fold at one epoch, keyed with `prefix`
    ('train'/'val'). Same definitions as _metrics."""
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
    """Trace `model`'s convergence (train AND val metrics per iteration) and
    return (history_rows, fitted_model) so the SAME estimator reports the metrics.

    Uses sklearn's callback API (SLEP023, sklearn >= 1.10) when available. Falls
    back to a warm_start refit trace on older versions.
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

    # Callback path (sklearn.callback, 1.10+), older versions ImportError into the
    # warm_start fallback below.
    try:
        from sklearn.callback import FitCallback  # type: ignore[import-not-found]

        class _MetricTracer(FitCallback):
            """Record the full metric set at each lbfgs iteration."""
            def __init__(self):
                self.rows = []
            def on_fit_task_begin(self, estimator, context, **kw):
                pass
            def on_fit_task_end(self, estimator, context, *, fitted_estimator=None,
                                **kw):
                clf = fitted_estimator if fitted_estimator is not None else estimator
                # only per-iteration subtasks carry a usable model
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
            # renumber sequentially in case the root 'fit' task fired too
            for i, r in enumerate(tracer.rows, start=1):
                r["iteration"] = i
            return tracer.rows, clf
        # nothing usable -> fall through to warm_start
    except ImportError:
        pass

    # Fallback: warm_start trace. Approximates the curve by refitting at
    # max_iter=1,2,3,... rather than observing the solver's own iterations.
    history = []
    if getattr(model, "warm_start", False):
        log.debug("sklearn.callback unavailable, tracing the training curve by "
                  "warm_start refit (max_iter=1..%d)", N_ITERS)
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
    """Stratification key: (vertex count, class)."""
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
    minority class is large enough, else plain random (ER has too few
    matching-wins to stratify safely)."""
    from sklearn.model_selection import train_test_split
    strat = _stratify_key(df)
    # Each stratum needs >= 2 to split, else stratify on label only, else not at all.
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


def _train_one_dataset(dataset, data_dir, feats):
    """Train logistic regression on one dataset. Returns a dict with history rows,
    the test-metrics row, held-out test predictions, and learned coefficients."""
    from sklearn.preprocessing import StandardScaler

    df, _ = _load_binary(data_dir, dataset)
    tr, val, test, dev = _split_three(df)
    log.info("%s: %d graphs (%d matching / %d pauli), "
             "split %d train / %d val / %d test", dataset, len(df),
             int(df.matching_wins.sum()), int((df.matching_wins == 0).sum()),
             len(tr), len(val), len(test))

    scaler = StandardScaler().fit(tr[feats].to_numpy(float))
    def X(frame):
        return scaler.transform(frame[feats].to_numpy(float))

    # Trace the SAME estimator's convergence on train vs val...
    history, _ = _train_history(
        _model(),
        X(tr), tr.matching_wins.to_numpy(), tr.delta_cx.to_numpy(float),
        X(val), val.matching_wins.to_numpy(), val.delta_cx.to_numpy(float))
    for row in history:                            # tag for grouping in plots
        row["dataset"] = dataset

    # ...then refit on train+val for the reported metrics.
    clf = _model().fit(X(dev), dev.matching_wins.to_numpy())
    y_true = test.matching_wins.to_numpy()
    test_row = _metrics("logistic", dataset, clf, X(test), y_true,
                        test.delta_cx.to_numpy(float))

    # Held-out test predictions (P(matching wins)), tagged with graph size for
    # the diagnostic plots.
    proba = clf.predict_proba(X(test))[:, 1]
    nv = test.n_vertices.to_numpy()
    preds = [{"dataset": dataset, "n_vertices": int(n),
              "y_true": int(t), "proba": round(float(p), 6)}
             for n, t, p in zip(nv, y_true, proba)]

    # Coefficients on standardized features, plus the scaler mean/scale per feature
    # so the decision function can be reconstructed on raw values downstream.
    coefs = [{"dataset": dataset, "feature": f, "coef": round(float(w), 6),
              "mean": round(float(m), 6), "scale": round(float(s), 6)}
             for f, w, m, s in zip(feats, clf.coef_.ravel(),
                                   scaler.mean_, scaler.scale_)]
    coefs.append({"dataset": dataset, "feature": "(intercept)",
                  "coef": round(float(clf.intercept_[0]), 6),
                  "mean": 0.0, "scale": 1.0})

    return {"history": history, "test_row": test_row,
            "preds": preds, "coefs": coefs}


def run(data_dir: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT) -> dict:
    """Train logistic regression separately on each dataset (er, structured,
    balanced) with its own train/val/test split. Writes training_history.csv
    (per-epoch train/val curves) and metrics.csv (held-out test metrics)."""
    feats = _model_features()
    all_history, all_test, all_preds, all_coefs = [], [], [], []
    for dataset in DATASETS:
        if not (data_dir / dataset / "labels" / "labels.csv").exists():
            log.warning("skipping '%s': no labels", dataset)
            continue
        r = _train_one_dataset(dataset, data_dir, feats)
        all_history.extend(r["history"])
        all_test.append(r["test_row"])
        all_preds.extend(r["preds"])
        all_coefs.extend(r["coefs"])

    # Put dataset first in the history CSV for readability.
    ordered = [{"dataset": r["dataset"], **{k: v for k, v in r.items()
                                          if k != "dataset"}}
               for r in all_history]
    _write_csv(out_dir / "training_history.csv", ordered)
    _write_csv(out_dir / "metrics.csv", all_test)
    _write_csv(out_dir / "test_predictions.csv", all_preds)
    _write_csv(out_dir / "coefficients.csv", all_coefs)

    # Feature values + signed cost gap per dataset, for the feature-vs-delta_cx
    # regression plot.
    import pandas as pd
    reg_frames = []
    for dataset in DATASETS:
        if (data_dir / dataset / "labels" / "labels.csv").exists():
            d, _ = _load_binary(data_dir, dataset)
            sub = d[feats + ["delta_cx"]].copy()
            sub.insert(0, "dataset", dataset)
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
        "datasets": [r["split"] for r in all_test],  # split == dataset name here
        "test_metrics": all_test,
        "out_dir": str(out_dir),
    }
