#!/usr/bin/env python3
"""Model training, tuning, and evaluation, dispatched by --experiment.

Experiments: train, tune, benchmark, ablation, overlap, lofo, per-size.

    python scripts/evaluate_models.py --experiment benchmark
    python scripts/evaluate_models.py --experiment ablation --ablation-mode subset
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from scipy.linalg import eigvalsh, expm
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from analysis.dataset_analysis import (  # noqa: E402
    DEFAULT_DATA,
    _model_features,
    load_dataset,
    set_ieee_style,
)
from model import evaluation as _bm  # noqa: E402
from progress import Progress, fmt_duration  # noqa: E402

log = logging.getLogger("overlap")
log_train = logging.getLogger("model")

FEATURES = _model_features()

# benchmark
DATASETS = _bm.DATASETS
SEEDS = _bm.SEEDS
OUT = _bm.OUT

# ablation
DATASET = "balanced"      # only dataset balanced enough to train on
NMAX = 256             # largest labeled size, bigger graphs are unlabeled
METRICS = ["accuracy", "precision", "recall", "specificity", "npv", "f1", "mcc",
           "cohen_kappa", "roc_auc", "pr_auc", "log_loss", "savings"]
DECOMP = ["pauli_num_terms", "match_num_matchings"]
# Tuned winner from hp_deep_tune.
ANN_KW = dict(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-4,
              learning_rate_init=1e-3, max_iter=1000, random_state=0)
SUBSET_METRICS = ["accuracy", "precision", "recall", "specificity", "npv", "f1",
                  "mcc", "roc_auc", "pr_auc"]
SUBSET_OUT = _ROOT / "results" / "evaluation" / "feature_subset_ablation.csv"

# overlap
DEFAULT_OUT = _ROOT / "results" / "analysis"

# lofo
FOLDS = [
    ("structured", "er"),
    ("er", "structured"),
]
LOFO_DATASETS = sorted({c for fold in FOLDS for c in fold})

# per-size
PER_SIZE_DECOMP = ["pauli_num_terms", "match_num_matchings"]
# Tuned winner from hp_deep_tune on the two decomposition counts.
PER_SIZE_ANN_KW = dict(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-4,
                       learning_rate_init=1e-3, max_iter=1000, random_state=0)

# tune
SWEEP_OUT = _ROOT / "results" / "evaluation" / "hp_sweep_mckay.csv"
DEEP_OUT = _ROOT / "results" / "evaluation" / "hp_deep_tune_mckay.csv"
ANN_SRC = _ROOT / "results" / "evaluation" / "hp_sweep_mckay.csv"

# Full metric suite collected per seed in deep mode.
DEEP_KEYS = ["accuracy", "precision", "recall", "specificity", "npv", "f1",
             "mcc", "roc_auc", "pr_auc", "savings"]


# =============================================================================
# benchmark
# =============================================================================
def run_benchmark(args) -> int:
    datasets = args.benchmark_dataset
    # Custom --dataset set writes a suffixed file so it never clobbers the canonical comparison.
    is_default = datasets == _bm.DATASETS
    out_name = ("model_comparison.csv" if is_default
                else f"model_comparison_{'_'.join(datasets)}.csv")

    _bm.OUT.mkdir(parents=True, exist_ok=True)
    n_models = len(_bm.learned_models())
    prog = Progress(total=len(datasets) * len(args.seeds) * n_models, unit="fit")
    print(f"fitting {prog.total} models "
          f"({n_models} models x {len(datasets)} datasets x {len(args.seeds)} seeds)")

    rows = []
    for dataset in datasets:
        df = _bm.load(dataset)
        per_model: dict[str, list[dict]] = {}

        for seed in args.seeds:
            train, _val, test = _bm.split(df, seed)
            for name, (estimator, needs_scale) in _bm.learned_models().items():
                per_model.setdefault(name, []).append(
                    _bm.score_model(clone(estimator), train, test, needs_scale))
                prog.step(f"{dataset}/{name} seed={seed}")

        for name, runs in per_model.items():
            agg = {"dataset": dataset, "model": name}
            for metric in _bm.METRICS:
                vals = np.array([r[metric] for r in runs], dtype=float)
                # A metric can be NaN across every seed by design (e.g. log-loss on SVM).
                # report NaN rather than let numpy warn about an empty slice.
                if np.isnan(vals).all():
                    agg[f"{metric}_mean"] = float("nan")
                    agg[f"{metric}_std"] = float("nan")
                else:
                    agg[f"{metric}_mean"] = np.nanmean(vals)
                    agg[f"{metric}_std"] = np.nanstd(vals)
            rows.append(agg)

    prog.done()

    ladder = pd.DataFrame(rows)
    ladder.to_csv(_bm.OUT / out_name, index=False)
    print(f"\nwrote {_bm.OUT / out_name}")

    # The confidence-interval file is balanced-specific, only produce it on the default run.
    if is_default:
        df = _bm.load("balanced")
        seed_rows = []
        for seed in args.seeds:
            train, _val, test = _bm.split(df, seed)
            row = _bm.score_model(LogisticRegression(max_iter=1000, random_state=0),
                                  train, test)
            row["seed"] = seed
            seed_rows.append(row)
        seeds = pd.DataFrame(seed_rows)
        seeds.to_csv(_bm.OUT / "seeds.csv", index=False)
        print(f"wrote {_bm.OUT / 'seeds.csv'}")
        print(f"\nshipped model across seeds: MCC {seeds.mcc.mean():.3f} "
              f"+/- {seeds.mcc.std():.3f}")

    # Per dataset, sorted by MCC, savings alongside to expose "good MCC, negative savings".
    for dataset in datasets:
        sub = ladder[ladder.dataset == dataset].sort_values("mcc_mean",
                                                           ascending=False)
        print(f"\n{dataset} model comparison (MCC, mean +/- std over {len(args.seeds)} seeds):")
        for _, r in sub.iterrows():
            print(f"  {r.model:<17} MCC {r.mcc_mean:6.3f} +/- {r.mcc_std:.3f}"
                  f"   savings {r.savings_mean:8.3f}")
    return 0


# =============================================================================
# ablation
# =============================================================================
# drop-one mode
def load(dataset: str) -> pd.DataFrame:
    """Features inner-joined to labels (labeled graphs only). ties dropped, y = 1[delta_cx > 0]."""
    d = _ROOT / "data" / dataset
    feat = pd.read_csv(d / "features" / "graph_features.csv")
    lab = pd.read_csv(d / "labels" / "labels.csv")

    df = feat.merge(lab[["graph_id", "delta_cx"]], on="graph_id", how="inner")
    df = df[(df.n_vertices <= NMAX) & (df.delta_cx != 0)].copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    return df


def split(df: pd.DataFrame, seed: int):
    """Leakage-safe ~64/16/20 split, stratified by (size, class) so no size shortcut wins."""
    def key(frame):
        k = frame.n_vertices.astype(str) + "_" + frame.y.astype(str)
        return k if k.value_counts().min() >= 2 else None

    dev, test = train_test_split(df, test_size=0.2, random_state=seed,
                                 stratify=key(df))
    train, val = train_test_split(dev, test_size=0.2, random_state=seed,
                                  stratify=key(dev))
    return train, val, test


def metrics(y_true, y_pred, dcx, score) -> dict:
    """The twelve metrics reported in results/logistic_model/metrics.csv."""
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) > 1

    # delta_cx = CX_pauli - CX_matching, positive means matching is cheaper.
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
    """Fit the shipped model on a feature subset, score on test. Scaler fit on train only."""
    scaler = StandardScaler().fit(train[features])
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(scaler.transform(train[features]), train.y)

    X_test = scaler.transform(test[features])
    return metrics(test.y.to_numpy(), clf.predict(X_test),
                   test.delta_cx.to_numpy(), clf.predict_proba(X_test)[:, 1])


def run_drop_one() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load(DATASET)

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
    # Flatten the ("mcc", "mean") MultiIndex to "mcc_mean".
    ablation.columns = [f"{m}_{stat}" for m, stat in ablation.columns]
    ablation = ablation.reset_index()

    # Rank by the drop in MCC when a feature is removed.
    full = ablation.loc[ablation.dropped == "(none)"].iloc[0]
    for metric in METRICS:
        ablation[f"{metric}_drop"] = full[f"{metric}_mean"] - ablation[f"{metric}_mean"]
    ablation = ablation.sort_values("mcc_drop", ascending=False)

    ablation.to_csv(OUT / "ablation.csv", index=False)
    print(f"wrote {OUT / 'ablation.csv'}")

    print(f"\ndrop-one-feature ablation on the {DATASET} dataset "
          f"({len(df)} graphs, mean over {len(SEEDS)} seeds):\n")
    print(f"  {'dropped':<22}{'MCC':>8}{'drop':>9}{'savings':>10}{'drop':>9}")
    for _, r in ablation.iterrows():
        name = "full model" if r.dropped == "(none)" else f"without {r.dropped}"
        drop_mcc = "" if r.dropped == "(none)" else f"{r.mcc_drop:+9.4f}"
        drop_sav = "" if r.dropped == "(none)" else f"{r.savings_drop:+9.4f}"
        print(f"  {name:<22}{r.mcc_mean:>8.4f}{drop_mcc:>9}"
              f"{r.savings_mean:>10.4f}{drop_sav:>9}")

    print("\nA near-zero drop means the feature carried no UNIQUE signal: the "
          "others already encode it.\nAll twelve metrics are in ablation.csv. MCC "
          "and CX savings are shown here.")
    return 0


# subset mode
def _threshold_rule(tr, te):
    """Best single threshold on n_Pauli, chosen on train, scored on test."""
    best_mcc, best = -2.0, None
    for t in np.unique(tr.pauli_num_terms):
        for low_is_match in (True, False):
            pred = ((tr.pauli_num_terms <= t) if low_is_match
                    else (tr.pauli_num_terms > t)).astype(int)
            m = matthews_corrcoef(tr.y, pred)
            if m > best_mcc:
                best_mcc, best = m, (t, low_is_match)
    t, low_is_match = best
    pred = ((te.pauli_num_terms <= t) if low_is_match
            else (te.pauli_num_terms > t)).astype(int)
    return pred


def run_subset(seeds) -> int:
    df = _bm.load("mckay")
    df = df[(df.n_vertices == 8) & (df.delta_cx != 0)].dropna(subset=DECOMP).copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    topo = list(_bm.FEATURES)

    subsets = {
        "topological only (10)": topo,
        "$n_{Pauli}$ only": ["pauli_num_terms"],
        "$n_{Pauli}$ + $n_{match}$": DECOMP,
        "$n_{Pauli}$ + degree variance": ["pauli_num_terms", "degree_variance"],
        "full (12)": topo + DECOMP,
    }

    rows = []
    for name, cols in subsets.items():
        acc = {k: [] for k in SUBSET_METRICS}
        for seed in seeds:
            tr, _v, te = _bm.split(df, seed)
            sc = StandardScaler().fit(tr[cols])
            clf = MLPClassifier(**ANN_KW).fit(sc.transform(tr[cols]), tr.y)
            pred = clf.predict(sc.transform(te[cols]))
            s = clf.predict_proba(sc.transform(te[cols]))[:, 1]
            m = _bm.metrics(te.y.to_numpy(), pred, te.delta_cx.to_numpy(), s)
            for k in SUBSET_METRICS:
                acc[k].append(m.get(k, float("nan")))
        row = {"subset": name, "n_features": len(cols)}
        for k in SUBSET_METRICS:
            row[k] = float(np.nanmean(acc[k]))
            row[f"{k}_std"] = float(np.nanstd(acc[k]))
        rows.append(row)
        print(f"  {name:32} MCC {row['mcc']:.3f} +- {row['mcc_std']:.3f}")

    # Non-learned baseline: one threshold on the leading feature.
    acc = {k: [] for k in SUBSET_METRICS}
    for seed in seeds:
        tr, _v, te = _bm.split(df, seed)
        pred = _threshold_rule(tr, te)
        m = _bm.metrics(te.y.to_numpy(), pred, te.delta_cx.to_numpy(), None)
        for k in SUBSET_METRICS:
            acc[k].append(m.get(k, float("nan")))
    row = {"subset": "threshold on $n_{Pauli}$", "n_features": 1}
    for k in SUBSET_METRICS:
        row[k] = float(np.nanmean(acc[k]))
        row[f"{k}_std"] = float(np.nanstd(acc[k]))
    rows.append(row)
    print(f"  {'threshold rule on n_Pauli':32} MCC {row['mcc']:.3f} "
          f"+- {row['mcc_std']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(SUBSET_OUT, index=False)
    print(f"\nwrote {SUBSET_OUT}")
    return 0


def run_ablation(args) -> int:
    if args.ablation_mode == "drop-one":
        return run_drop_one()
    return run_subset(args.seeds)


# =============================================================================
# overlap
# =============================================================================
def _standardize(X):
    """z-score each column, guarding zero-variance (constant within a size) features."""
    import numpy as np
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def overlap_within_size(df, dataset, k):
    """One row per (dataset, vertex count) with both classes. k-NN disagreement and NN
    geometry on standardized features, within each size so a size shortcut can't pose as
    class separation."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    feats = _model_features()
    rows = []
    for n, grp in df.groupby("n_vertices"):
        y = (grp.delta_cx > 0).to_numpy().astype(int)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            # Single-class size: nothing to overlap.
            rows.append(dict(dataset=dataset, n_vertices=int(n), n_graphs=len(grp),
                             n_pos=n_pos, n_neg=n_neg, single_class=True,
                             knn_disagree_pos=float("nan"),
                             knn_disagree_neg=float("nan"),
                             pos_nn_is_neg=float("nan"),
                             collide_frac=float("nan"),
                             embedded_frac=float("nan"),
                             sep_ratio_med=float("nan")))
            continue

        X = _standardize(grp[feats].to_numpy(dtype=float))

        # k+1 neighbours: the first is the point itself, dropped below.
        kk = min(k + 1, len(X))
        nn = NearestNeighbors(n_neighbors=kk).fit(X)
        dist, idx = nn.kneighbors(X)
        idx, dist = idx[:, 1:], dist[:, 1:]           # drop self
        neigh_y = y[idx]

        disagree = (neigh_y != y[:, None]).mean(axis=1)
        knn_dis_pos = float(disagree[y == 1].mean())
        knn_dis_neg = float(disagree[y == 0].mean())

        # For each positive: is its single nearest neighbour a negative?
        pos_mask = y == 1
        nearest_label = neigh_y[:, 0]
        pos_nn_is_neg = float((nearest_label[pos_mask] == 0).mean())

        # Per-positive nearest same-label vs opposite-label distance, over the full
        # within-size set. collide_frac = identical negative, embedded_frac = inside
        # the negative cloud, sep_ratio_med = median distance ratio.
        from scipy.spatial.distance import cdist
        Xp = X[pos_mask]
        d_same = cdist(Xp, Xp)
        np.fill_diagonal(d_same, np.inf)              # exclude self
        nearest_same = d_same.min(axis=1)             # inf if only one positive
        nearest_opp = cdist(Xp, X[y == 0]).min(axis=1)

        collide_frac = float((nearest_opp == 0).mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(nearest_opp > 0,
                              nearest_same / nearest_opp, np.inf)
        have_peer = np.isfinite(nearest_same)         # need >= 2 positives
        embedded_frac = (float((ratios[have_peer] >= 1).mean())
                         if have_peer.any() else float("nan"))
        finite = have_peer & np.isfinite(ratios)
        sep_ratio_med = (float(np.median(ratios[finite]))
                         if finite.any() else float("nan"))

        rows.append(dict(dataset=dataset, n_vertices=int(n), n_graphs=len(grp),
                         n_pos=n_pos, n_neg=n_neg, single_class=False,
                         knn_disagree_pos=knn_dis_pos,
                         knn_disagree_neg=knn_dis_neg,
                         pos_nn_is_neg=pos_nn_is_neg,
                         collide_frac=collide_frac,
                         embedded_frac=embedded_frac,
                         sep_ratio_med=sep_ratio_med))
    return rows


def plot_embedding(dfs, out_dir):
    """2-D PCA embedding of the mixed-class size per dataset, positives on top. if they
    sit inside the negative cloud, no boundary in this space separates them."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    set_ieee_style()
    feats = _model_features()
    datasets = list(dfs)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.2),
                             squeeze=False)

    for ax, dataset in zip(axes[0], datasets):
        df = dfs[dataset]
        y = (df.delta_cx > 0).to_numpy().astype(int)
        # Visualize the mixed-class size with the most positives.
        mixed = [n for n, g in df.groupby("n_vertices")
                 if (g.delta_cx > 0).any() and (g.delta_cx <= 0).any()]
        if not mixed:
            ax.set_title(f"{dataset}: no mixed size")
            ax.axis("off")
            continue
        best = max(mixed, key=lambda n:
                   int((df[df.n_vertices == n].delta_cx > 0).sum()))
        sub = df[df.n_vertices == best]
        ys = (sub.delta_cx > 0).to_numpy().astype(int)
        X = _standardize(sub[feats].to_numpy(dtype=float))
        Z = PCA(n_components=2, random_state=0).fit_transform(X)

        ax.scatter(Z[ys == 0, 0], Z[ys == 0, 1], s=6, c="0.7",
                   label=f"Pauli ({int((ys == 0).sum())})", rasterized=True)
        ax.scatter(Z[ys == 1, 0], Z[ys == 1, 1], s=28, c="crimson",
                   marker="X", edgecolors="black", linewidths=0.4,
                   label=f"matching ({int(ys.sum())})")
        ax.set_title(f"{dataset}, n={best}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize="small")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "class_overlap_embedding.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    log.info("wrote %s", pdf)


def _print_report(rows):
    """Per-size overlap table plus a short reading of it."""
    print("\nWithin-size class overlap (only sizes with BOTH classes carry a "
          "decision):\n")
    print(f"  {'dataset':<12}{'n':>5}{'pos':>6}{'neg':>7}"
          f"{'disagree+':>11}{'collide':>9}{'embedded':>10}{'sep_med':>9}")
    for r in rows:
        if r["single_class"]:
            print(f"  {r['dataset']:<12}{r['n_vertices']:>5}{r['n_pos']:>6}"
                  f"{r['n_neg']:>7}   single class (no decision)")
            continue
        sep = r["sep_ratio_med"]
        sep_s = "     inf" if sep == float("inf") else f"{sep:>9.3f}"
        print(f"  {r['dataset']:<12}{r['n_vertices']:>5}{r['n_pos']:>6}"
              f"{r['n_neg']:>7}{r['knn_disagree_pos']:>11.3f}"
              f"{r['collide_frac']:>9.3f}{r['embedded_frac']:>10.3f}{sep_s}")

    print("\nHow to read this:")
    print("  disagree+  fraction of a positive's k nearest neighbours that are")
    print("             negative. Near 1.0: the positives are surrounded by")
    print("             negatives, so Bayes error is high here. No boundary captures")
    print("             them without also flipping the negatives.")
    print("  collide    fraction of positives that share IDENTICAL features with some")
    print("             negative. These cannot be separated by ANY model reading these")
    print("             features: the inputs are the same, the labels differ.")
    print("  embedded   fraction of positives whose nearest negative is at least as")
    print("             close as their nearest fellow positive (ratio >= 1): they sit")
    print("             inside the negative cloud.")
    print("  sep_med    median nearest-same / nearest-opposite distance. < 1: positives")
    print("             cluster (separable). > 1: dispersed among negatives.")
    print("\nWhere disagree+, collide and embedded are all high, the classes are not")
    print("separable from these features, and no amount of model tuning recovers")
    print("signal the feature space does not contain.")


def run_overlap(args) -> int:
    t0 = time.perf_counter()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO)
    for noisy in ("matplotlib", "matplotlib.font_manager", "fontTools",
                  "fontTools.subset", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import pandas as pd

    all_rows, dfs = [], {}
    for dataset in args.overlap_dataset:
        df = load_dataset(args.overlap_data, dataset)
        df = df[df.delta_cx != 0].copy()          # drop ties, matching training
        dfs[dataset] = df
        all_rows.extend(overlap_within_size(df, dataset, args.overlap_k))

    _print_report(all_rows)

    args.overlap_out.mkdir(parents=True, exist_ok=True)
    csv = args.overlap_out / "class_overlap.csv"
    pd.DataFrame(all_rows).to_csv(csv, index=False)
    print(f"\nwrote {csv}")

    if not args.overlap_no_plots:
        plot_embedding(dfs, args.overlap_out)

    print(f"  done in {fmt_duration(time.perf_counter() - t0)}")
    return 0


# =============================================================================
# lofo
# =============================================================================
def lofo_load(dataset: str) -> pd.DataFrame:
    """Features inner-joined to labels (labeled graphs only). ties dropped, y = 1[delta_cx > 0]."""
    d = _ROOT / "data" / dataset
    feat = pd.read_csv(d / "features" / "graph_features.csv")
    lab = pd.read_csv(d / "labels" / "labels.csv")

    df = feat.merge(lab[["graph_id", "delta_cx"]], on="graph_id", how="inner")
    df = df[(df.n_vertices <= NMAX) & (df.delta_cx != 0)].copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    return df


def lofo_metrics(y_true, y_pred, dcx, score) -> dict:
    """The twelve metrics in results/logistic_model/metrics.csv. read MCC and savings, not
    accuracy (a constant predictor scores 99.8% on ER)."""
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) > 1

    # delta_cx = CX_pauli - CX_matching, positive means matching is cheaper.
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


def run_lofo(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = {c: lofo_load(c) for c in LOFO_DATASETS}

    prog = Progress(total=len(FOLDS) * len(SEEDS), unit="fit")
    print(f"fitting {prog.total} models "
          f"({len(FOLDS)} folds x {len(SEEDS)} seeds)")

    rows = []
    for train_on, test_on in FOLDS:
        train_df, test_df = datasets[train_on], datasets[test_on]

        for seed in SEEDS:
            # Scaler fit on train rows only, so nothing leaks.
            scaler = StandardScaler().fit(train_df[FEATURES])
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(scaler.transform(train_df[FEATURES]), train_df.y)

            X_test = scaler.transform(test_df[FEATURES])
            row = lofo_metrics(test_df.y.to_numpy(), clf.predict(X_test),
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

    print(f"\n  Compare with MCC ~0.96 on the balanced (results/logistic_model/metrics.csv), "
          "where the\n  model may use the family fingerprint. Falling to "
          f"{summary.mcc.min():.3f} means it does not\n  transfer: the features that "
          "separate the classes within a family partly encode\n  WHICH family the "
          "graph came from, so holding a family out strips away a cue\n  the model "
          "was leaning on.")
    return 0


# =============================================================================
# per-size
# =============================================================================
def run_per_size(args) -> int:
    out = _ROOT / "results" / "evaluation" / f"ann_per_size_{args.per_size_dataset}.csv"
    df = _bm.load(args.per_size_dataset)
    df = df[df.delta_cx != 0].dropna(subset=PER_SIZE_DECOMP + ["pauli_num_terms"]).copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    cols = list(_bm.FEATURES) + PER_SIZE_DECOMP
    # Only sizes with decomposition features remain, evaluate each as its own sub-dataset.
    SIZES = sorted(int(n) for n in df.n_vertices.unique())

    rows = []
    for n in SIZES:
        sub = df[df.n_vertices == n]
        pos = int(sub.y.sum())
        single = pos == 0 or pos == len(sub)
        # Full metric suite per seed, matching the population-level tables.
        keys = ["accuracy", "precision", "recall", "specificity", "npv", "f1",
                "mcc", "roc_auc", "pr_auc", "savings"]
        acc = {k: [] for k in keys}
        for seed in args.seeds:
            # Stratify only when both classes are present.
            strat = sub.y if not single else None
            tr, te = train_test_split(sub, test_size=0.2, random_state=seed,
                                      stratify=strat)
            sc = StandardScaler().fit(tr[cols])
            Xtr, Xte = sc.transform(tr[cols]), sc.transform(te[cols])
            clf = MLPClassifier(**PER_SIZE_ANN_KW)
            if tr.y.nunique() < 2:
                # Single-class training set: predict that class.
                pred = np.full(len(te), int(tr.y.iloc[0]))
                proba = pred.astype(float)
            else:
                clf.fit(Xtr, tr.y)
                pred = clf.predict(Xte)
                proba = clf.predict_proba(Xte)[:, 1]
            m = _bm.metrics(te.y.to_numpy(), pred, te.delta_cx.to_numpy(), proba)
            for k in keys:
                acc[k].append(m.get(k, float("nan")))
        row = {"n_vertices": n, "n_graphs": len(sub), "matching_wins": pos,
               "single_class": single, "accuracy": float(np.mean(acc["accuracy"])),
               "accuracy_std": float(np.std(acc["accuracy"]))}
        # Class-conditional metrics are undefined for a single-class size.
        for k in keys:
            if k == "accuracy":
                continue
            row[k] = (float("nan") if single else float(np.nanmean(acc[k])))
            row[f"{k}_std"] = (float("nan") if single else float(np.nanstd(acc[k])))
        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    print(f"wrote {out}\n")
    print(f"{'n':>5}{'graphs':>8}{'wins':>6}{'accuracy':>10}{'MCC':>8}"
          f"{'recall':>8}{'prec':>8}{'savings':>10}")
    for _, r in res.iterrows():
        mcc = "     n/a" if r.single_class else f"{r.mcc:>8.3f}"
        rec = "     n/a" if r.single_class else f"{r.recall:>8.3f}"
        prc = "     n/a" if r.single_class else f"{r.precision:>8.3f}"
        # savings undefined for a single-class size.
        sav = "       n/a" if r.single_class else f"{r.savings:>10.3f}"
        tag = "  (single class)" if r.single_class else ""
        print(f"{int(r.n_vertices):>5}{int(r.n_graphs):>8}{int(r.matching_wins):>6}"
              f"{r.accuracy:>10.3f}{mcc}{rec}{prc}{sav}{tag}")
    return 0


# =============================================================================
# train
# =============================================================================
# logistic mode
def _print_report_logistic(report: dict) -> None:
    print(f"\n{report['n_features']} features, target = {report['target']}")
    print("\nheld-out TEST metrics (logistic regression), per dataset:")
    show = ["accuracy", "f1", "mcc", "roc_auc", "pr_auc", "savings_captured"]
    hdr = "  {:<12}".format("dataset") + "".join(f"{m[:9]:>11}" for m in show)
    print(hdr)
    for r in report["test_metrics"]:
        line = "  {:<12}".format(r["split"])
        line += "".join(f"{str(r.get(m, '')):>11}" for m in show)
        print(line)


def run_train_logistic(args) -> int:
    if not (args.train_data / "balanced" / "labels" / "labels.csv").exists():
        log_train.error("missing balanced dataset. Run `scripts/build_balanced.py` first.")
        return 2

    try:
        import pandas, sklearn  # noqa: F401
    except ImportError as e:
        log_train.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    from model.logistic import DEFAULT_OUT, run  # noqa: E402

    out = args.train_out if args.train_out is not None else DEFAULT_OUT
    report = run(args.train_data, out)
    _print_report_logistic(report)

    if not args.train_no_plots:
        from model.plots import plot_all
        plot_all(out)

    print(f"\nmetrics + history + plots -> {out}")
    return 0


# =============================================================================
# tune
# =============================================================================
def _load_population(dataset):
    """Load/filter the population, add the binary label. Returns (df, cols)."""
    df = _bm.load(dataset)
    df = df[(df.n_vertices == 8) & (df.delta_cx != 0)].dropna(subset=DECOMP).copy()
    df["y"] = (df.delta_cx > 0).astype(int)
    cols = list(_bm.FEATURES) + DECOMP
    return df, cols


# sweep mode
def _search_space():
    """{name: (estimator, param_grid, needs_scaling)}. Modest grids for tractability."""
    return {
        "logistic": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                     {"C": [0.1, 1.0, 10.0]}, True),
        "naive_bayes": (GaussianNB(), {"var_smoothing": [1e-9, 1e-7, 1e-5]}, True),
        "dtree": (DecisionTreeClassifier(class_weight="balanced", random_state=0),
                  {"max_depth": [1, 2, 3, 5, 8, None],
                   "min_samples_leaf": [1, 5, 20]}, False),
        "knn": (KNeighborsClassifier(),
                {"n_neighbors": [3, 5, 11, 21], "weights": ["uniform", "distance"]},
                True),
        "svm_rbf": (SVC(probability=True, class_weight="balanced", random_state=0),
                    {"C": [1.0, 10.0], "gamma": ["scale", 0.1]}, True),
        "mlp": (MLPClassifier(max_iter=1000, random_state=0),
                {"hidden_layer_sizes": [(32,), (64,), (128,), (256,),
                                        (64, 32), (128, 64), (256, 128),
                                        (128, 64, 32), (256, 128, 64)],
                 "alpha": [1e-4, 1e-3, 1e-2],
                 "learning_rate_init": [1e-3, 1e-2],
                 "activation": ["relu", "tanh"]}, True),
        "rand_forest": (RandomForestClassifier(class_weight="balanced",
                                               random_state=0),
                        {"n_estimators": [200, 400],
                         "max_depth": [None, 8, 16],
                         "min_samples_leaf": [1, 5]}, False),
        "grad_boost": (GradientBoostingClassifier(random_state=0),
                       {"n_estimators": [200, 400], "max_depth": [2, 3],
                        "learning_rate": [0.05, 0.1]}, False),
        "xgboost": (XGBClassifier(eval_metric="logloss", verbosity=0,
                                  random_state=0),
                    {"n_estimators": [200, 400], "max_depth": [3, 6],
                     "learning_rate": [0.05, 0.1],
                     "scale_pos_weight": [1, 25]}, False),
        "lightgbm": (LGBMClassifier(verbose=-1, random_state=0),
                     {"n_estimators": [200, 400], "num_leaves": [15, 31],
                      "learning_rate": [0.05, 0.1],
                      "class_weight": [None, "balanced"]}, False),
    }


def run_tune_sweep(args) -> int:
    df, cols = _load_population(args.tune_dataset)
    print(f"{args.tune_dataset} (Brandon McKay): {len(df)} graphs, "
          f"{int(df.y.sum())} positives, "
          f"{len(cols)} features (classical + decomposition)")

    space = _search_space()
    prog = Progress(total=len(space), unit="model")
    rows = []
    for name, (est, grid, scale) in space.items():
        # Tune on seed 0's train split (5-fold CV, MCC), then evaluate the best estimator
        # across all seeds' test splits.
        tr0, _v, _te = _bm.split(df, 0)
        Xtr0 = tr0[cols]
        if scale:
            sc0 = StandardScaler().fit(Xtr0)
            Xtr0 = sc0.transform(Xtr0)
        else:
            Xtr0 = Xtr0.to_numpy()
        gs = GridSearchCV(est, grid, scoring="matthews_corrcoef", cv=5, n_jobs=-1)
        gs.fit(Xtr0, tr0.y)
        best_params = gs.best_params_

        mccs, savs, recs, precs = [], [], [], []
        for seed in args.seeds:
            tr, _v, te = _bm.split(df, seed)
            Xtr, Xte = tr[cols], te[cols]
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            else:
                Xtr, Xte = Xtr.to_numpy(), Xte.to_numpy()
            clf = est.__class__(**{**est.get_params(), **best_params})
            clf.fit(Xtr, tr.y)
            pred = clf.predict(Xte)
            s = (clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba")
                 else clf.decision_function(Xte)
                 if hasattr(clf, "decision_function") else None)
            m = _bm.metrics(te.y.to_numpy(), pred, te.delta_cx.to_numpy(), s)
            mccs.append(m["mcc"]); savs.append(m["savings"])
            recs.append(m["recall"]); precs.append(m["precision"])
        rows.append({
            "model": name, "best_params": str(best_params),
            "mcc": float(np.mean(mccs)), "mcc_std": float(np.std(mccs)),
            "recall": float(np.mean(recs)), "precision": float(np.mean(precs)),
            "savings": float(np.nanmean(savs)),
        })
        prog.step(name)
    prog.done()

    res = pd.DataFrame(rows).sort_values("mcc", ascending=False).reset_index(drop=True)
    res.to_csv(SWEEP_OUT, index=False)
    print(f"\nwrote {SWEEP_OUT}\n")
    print(f"{'model':<14}{'MCC':>8}{'recall':>9}{'prec':>8}{'savings':>10}"
          f"   best params")
    for _, r in res.iterrows():
        print(f"  {r.model:<12}{r.mcc:>8.3f}{r.recall:>9.3f}{r.precision:>8.3f}"
              f"{r.savings:>10.3f}   {r.best_params}")
    win = res.iloc[0]
    print(f"\nWinner by MCC: {win.model} (MCC {win.mcc:.3f}, savings {win.savings:.3f})")
    return 0


# deep mode
def _deep_grids():
    """A much larger grid per model, ordered cheap-to-expensive so fast models finish first."""
    return {
        "naive_bayes": (GaussianNB(),
                        {"var_smoothing": np.logspace(-11, -3, 9)}, True),
        "logistic": (LogisticRegression(max_iter=5000, class_weight="balanced"),
                     {"C": np.logspace(-3, 3, 13),
                      "penalty": ["l2"], "solver": ["lbfgs"]}, True),
        "dtree": (DecisionTreeClassifier(class_weight="balanced", random_state=0),
                  {"max_depth": [1, 2, 3, 4, 5, 6, 8, 12, None],
                   "min_samples_leaf": [1, 2, 5, 10, 20, 50],
                   "criterion": ["gini", "entropy"]}, False),
        "knn": (KNeighborsClassifier(),
                {"n_neighbors": [1, 3, 5, 7, 11, 15, 21, 31],
                 "weights": ["uniform", "distance"],
                 "p": [1, 2]}, True),
        "svm_rbf": (SVC(probability=True, class_weight="balanced", random_state=0),
                    {"C": [0.1, 1.0, 10.0, 100.0],
                     "gamma": ["scale", "auto", 0.01, 0.1, 1.0]}, True),
        "rand_forest": (RandomForestClassifier(class_weight="balanced",
                                               random_state=0, n_jobs=-1),
                        {"n_estimators": [200, 400, 800],
                         "max_depth": [None, 6, 12, 20],
                         "min_samples_leaf": [1, 2, 5, 10],
                         "max_features": ["sqrt", "log2", None]}, False),
        "grad_boost": (GradientBoostingClassifier(random_state=0),
                       {"n_estimators": [200, 400, 800],
                        "max_depth": [2, 3, 4],
                        "learning_rate": [0.02, 0.05, 0.1],
                        "subsample": [0.8, 1.0]}, False),
        "lightgbm": (LGBMClassifier(verbose=-1, random_state=0, n_jobs=-1),
                     {"n_estimators": [200, 400, 800],
                      "num_leaves": [15, 31, 63],
                      "learning_rate": [0.02, 0.05, 0.1],
                      "class_weight": [None, "balanced"],
                      "min_child_samples": [5, 20]}, False),
        "xgboost": (XGBClassifier(eval_metric="logloss", verbosity=0,
                                  random_state=0, n_jobs=-1),
                    {"n_estimators": [200, 400, 800],
                     "max_depth": [3, 4, 6],
                     "learning_rate": [0.02, 0.05, 0.1],
                     "scale_pos_weight": [1, 10, 25],
                     "subsample": [0.8, 1.0]}, False),
    }


def _tune_one(name, est, grid, scale, df, cols, seeds):
    """GridSearchCV on seed 0, evaluate best across all seeds. Returns a result dict."""
    tr0, _v, _te = _bm.split(df, 0)
    Xtr0 = tr0[cols]
    if scale:
        Xtr0 = StandardScaler().fit_transform(Xtr0)
    else:
        Xtr0 = Xtr0.to_numpy()
    gs = GridSearchCV(est, grid, scoring="matthews_corrcoef", cv=5, n_jobs=-1)
    gs.fit(Xtr0, tr0.y)
    best = gs.best_params_

    # Full metric suite per seed, then average.
    keys = DEEP_KEYS
    acc = {k: [] for k in keys}
    for seed in seeds:
        tr, _v, te = _bm.split(df, seed)
        Xtr, Xte = tr[cols], te[cols]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        else:
            Xtr, Xte = Xtr.to_numpy(), Xte.to_numpy()
        clf = est.__class__(**{**est.get_params(), **best})
        clf.fit(Xtr, tr.y)
        pred = clf.predict(Xte)
        s = (clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba")
             else clf.decision_function(Xte)
             if hasattr(clf, "decision_function") else None)
        m = _bm.metrics(te.y.to_numpy(), pred, te.delta_cx.to_numpy(), s)
        for k in keys:
            acc[k].append(m.get(k, float("nan")))
    out = {"model": name, "best_params": str(best),
           "n_configs": len(gs.cv_results_["params"]),
           "mcc_std": float(np.std(acc["mcc"]))}
    for k in keys:
        out[k] = float(np.nanmean(acc[k]))
        out[f"{k}_std"] = float(np.nanstd(acc[k]))
    return out


def run_tune_deep(args) -> int:
    df, cols = _load_population(args.tune_dataset)
    print(f"{args.tune_dataset} (Brandon McKay): {len(df)} graphs, "
          f"{int(df.y.sum())} positives, {len(cols)} features")

    # Resume: seed output with the preserved ANN row and any models tuned in a prior run.
    done = {}
    if DEEP_OUT.exists():
        for _, r in pd.read_csv(DEEP_OUT).iterrows():
            done[r["model"]] = r.to_dict()
    elif ANN_SRC.exists():
        # Recompute the ANN row at its winning config (32-node layer, tanh, alpha 1e-4) so
        # it carries the same full metric suite as the deep-tuned models.
        ann_kw = dict(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-4,
                      learning_rate_init=1e-3, max_iter=1000, random_state=0)
        keys = DEEP_KEYS
        acc = {k: [] for k in keys}
        for seed in args.seeds:
            tr, _v, te = _bm.split(df, seed)
            sc = StandardScaler().fit(tr[cols])
            Xtr, Xte = sc.transform(tr[cols]), sc.transform(te[cols])
            clf = MLPClassifier(**ann_kw).fit(Xtr, tr.y)
            pred = clf.predict(Xte)
            s = clf.predict_proba(Xte)[:, 1]
            m = _bm.metrics(te.y.to_numpy(), pred, te.delta_cx.to_numpy(), s)
            for k in keys:
                acc[k].append(m.get(k, float("nan")))
        row = {"model": "mlp", "best_params": str(ann_kw), "n_configs": 108,
               "mcc_std": float(np.std(acc["mcc"]))}
        for k in keys:
            row[k] = float(np.nanmean(acc[k]))
            row[f"{k}_std"] = float(np.nanstd(acc[k]))
        done["mlp"] = row

    grids = _deep_grids()
    todo = [m for m in grids if (args.tune_only is None or m in args.tune_only)
            and m not in done]
    print(f"tuning {len(todo)} models (one at a time): {', '.join(todo)}")
    if "mlp" in done:
        print("  (ANN preserved from the earlier expanded search)")

    for i, name in enumerate(todo, 1):
        est, grid, scale = grids[name]
        n_cfg = int(np.prod([len(v) for v in grid.values()]))
        t0 = time.perf_counter()
        print(f"\n[{i}/{len(todo)}] {name}: grid of {n_cfg} configs ...", flush=True)
        res = _tune_one(name, est, grid, scale, df, cols, args.seeds)
        done[name] = res
        # Save after every model so nothing is lost on a long run.
        out = pd.DataFrame(list(done.values())).sort_values(
            "mcc", ascending=False).reset_index(drop=True)
        out.to_csv(DEEP_OUT, index=False)
        print(f"    MCC {res['mcc']:.3f}  savings {res['savings']:+.3f}  "
              f"({fmt_duration(time.perf_counter() - t0)})  -> saved")

    out = pd.DataFrame(list(done.values())).sort_values(
        "mcc", ascending=False).reset_index(drop=True)
    out.to_csv(DEEP_OUT, index=False)
    print(f"\nwrote {DEEP_OUT}\n")
    print(f"{'model':<14}{'MCC':>8}{'recall':>9}{'prec':>8}{'savings':>10}  best params")
    for _, r in out.iterrows():
        print(f"  {r.model:<12}{r.mcc:>8.3f}{r.recall:>9.3f}{r.precision:>8.3f}"
              f"{r.savings:>10.3f}  {r.best_params}")
    win = out.iloc[0]
    print(f"\nWinner by MCC: {win.model} (MCC {win.mcc:.3f}, savings {win.savings:.3f})")
    return 0


# =============================================================================
# CLI
# =============================================================================
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="scripts/evaluate_models.py",
        description="Model training, tuning, and evaluation, dispatched by "
                    "--experiment.")
    p.add_argument("--experiment", required=True,
                   choices=["train", "tune", "benchmark", "ablation",
                            "overlap", "lofo", "per-size"],
                   help="which experiment to run")

    # shared
    p.add_argument("--seeds", type=int, nargs="+", default=_bm.SEEDS,
                   help=f"seeds (default: {' '.join(map(str, _bm.SEEDS))})")

    # train
    p.add_argument("--train-data", type=Path, default=DEFAULT_DATA,
                   help=f"[train] data root (default: {DEFAULT_DATA})")
    p.add_argument("--train-out", type=Path, default=None,
                   help="[train] output dir (default: results/logistic_model)")
    p.add_argument("--train-no-plots", action="store_true",
                   help="[train] skip building the PDF plots")
    p.add_argument("--train-dataset", nargs="+", default=["er", "structured"],
                   choices=["er", "structured", "mckay"],
                   help="[train] datasets to evaluate (default: er structured). "
                        "mckay is the exhaustive set of all 11,117 connected "
                        "8-vertex graphs")

    # tune
    p.add_argument("--tune-mode", choices=["sweep", "deep"], default="deep",
                   help="[tune] sweep = modest grids over all models (incl. ANN). "
                        "deep = large per-model grids, ANN preserved from the sweep CSV "
                        "(default: deep)")
    p.add_argument("--tune-dataset", default="mckay",
                   help="[tune] graph population to load (default: mckay)")
    p.add_argument("--tune-only", nargs="+", default=None,
                   help="[tune] deep mode only: tune only these models "
                        "(default: all non-ANN)")

    # benchmark
    p.add_argument("--benchmark-dataset", nargs="+", default=_bm.DATASETS,
                   help=f"[benchmark] datasets to evaluate "
                        f"(default: {' '.join(_bm.DATASETS)}). mckay is the "
                        "exhaustive set of all 11,117 connected 8-vertex graphs.")

    # ablation
    p.add_argument("--ablation-mode", choices=["drop-one", "subset"],
                   default="subset",
                   help="[ablation] drop-one: drop-each-feature on the shipped "
                        "logistic model (balanced dataset). subset (default): "
                        "nested feature subsets for the tuned ANN on mckay.")

    # overlap
    p.add_argument("--overlap-dataset", nargs="+", default=["er", "structured"],
                   choices=["er", "structured", "balanced", "mckay"],
                   help="[overlap] datasets to analyze (default: er structured). "
                        "mckay is the exhaustive set of all 11,117 connected "
                        "8-vertex graphs")
    p.add_argument("--overlap-data", type=Path, default=DEFAULT_DATA,
                   help=f"[overlap] data root (default: {DEFAULT_DATA})")
    p.add_argument("--overlap-out", type=Path, default=DEFAULT_OUT,
                   help=f"[overlap] output dir (default: {DEFAULT_OUT})")
    p.add_argument("--overlap-k", type=int, default=10,
                   help="[overlap] neighbours for the k-NN disagreement (default: 10)")
    p.add_argument("--overlap-no-plots", action="store_true",
                   help="[overlap] skip the embedding plot")
    p.add_argument("-v", "--verbose", action="store_true")

    # per-size
    p.add_argument("--per-size-dataset", default="er",
                   help="[per-size] er or structured (default: er)")

    args = p.parse_args(list(argv) if argv is not None else None)

    if args.experiment == "train":
        return run_train_logistic(args)
    if args.experiment == "tune":
        return (run_tune_sweep(args) if args.tune_mode == "sweep"
                else run_tune_deep(args))
    if args.experiment == "benchmark":
        return run_benchmark(args)
    if args.experiment == "ablation":
        return run_ablation(args)
    if args.experiment == "overlap":
        return run_overlap(args)
    if args.experiment == "lofo":
        return run_lofo(args)
    return run_per_size(args)


if __name__ == "__main__":
    sys.exit(main())
