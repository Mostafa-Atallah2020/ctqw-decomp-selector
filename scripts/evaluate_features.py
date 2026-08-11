#!/usr/bin/env python3
"""Feature-set evaluation experiments for the decomposition selector (CLI).

Experiments routed by --mode: features, allmetrics (default), spectral.
--dataset defaults to mckay (the tested path for features/allmetrics, spectral
supports other datasets via --dataset/--all-sizes).

Examples:
    python scripts/evaluate_features.py --mode allmetrics
    python scripts/evaluate_features.py --mode spectral --dataset er --all-sizes
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from scipy.linalg import eigvalsh, expm

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dataset.features import spectral_rich, pauli_structure  # noqa: E402
from model import evaluation as _bm  # noqa: E402

from sklearn.base import clone  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from progress import Progress  # noqa: E402

# Shared constants.
OUT = _ROOT / "results" / "evaluation"
SEEDS = [0, 1, 2, 3, 4]

# allmetrics mode constants
METRICS = ["accuracy", "precision", "recall", "specificity", "npv", "f1", "mcc",
           "roc_auc", "pr_auc", "savings"]
ALLMETRICS_OUT = _ROOT / "results" / "evaluation" / "enrich_allmetrics_mckay.csv"


# features mode
def _score_set_features(df, cols, seeds):
    """Model comparison (twelve classifiers) on a column set. Returns best
    (model, mcc, savings) over
    seeds and the per-model means."""
    per_model = {}
    for name, (est, needs_scale) in _bm.learned_models().items():
        mccs, savs = [], []
        for seed in seeds:
            tr, _v, te = _bm.split(df, seed)
            if needs_scale:
                sc = StandardScaler().fit(tr[cols])
                Xtr, Xte = sc.transform(tr[cols]), sc.transform(te[cols])
            else:
                Xtr, Xte = tr[cols], te[cols]
            clf = clone(est).fit(Xtr, tr.y)
            pred = clf.predict(Xte)
            s = (clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba")
                 else clf.decision_function(Xte)
                 if hasattr(clf, "decision_function") else None)
            m = _bm.metrics(te.y.to_numpy(), pred, te.delta_cx.to_numpy(), s)
            mccs.append(m["mcc"])
            savs.append(m["savings"])
        per_model[name] = (float(np.mean(mccs)), float(np.nanmean(savs)))
    best = max(per_model, key=lambda k: per_model[k][0])
    return best, per_model[best], per_model


def run_features(args) -> int:
    df = _bm.load(args.dataset)
    n_mode = int(df.n_vertices.mode().iloc[0])
    df = df[df.n_vertices == n_mode].reset_index(drop=True)

    g6 = (_ROOT / "data" / args.dataset / "g6" / "all.g6").read_text().split()
    feat_full = pd.read_csv(_ROOT / "data" / args.dataset / "features"
                            / "graph_features.csv").reset_index(drop=True)
    g6_by_id = dict(zip(feat_full["graph_id"], g6[: len(feat_full)]))

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=0).reset_index(
            drop=True)
    print(f"{args.dataset}: {len(df)} graphs at n={n_mode}, "
          f"{int(df.y.sum())} matching-wins ({100 * df.y.mean():.2f}%)")

    print("computing spectral-rich and pauli-structure features...")
    prog = Progress(total=len(df), unit="graph")
    sr, ps = [], []
    for gid in df.graph_id:
        g6str = g6_by_id[gid]
        G = nx.from_graph6_bytes(g6str.encode())
        sr.append(spectral_rich(G))
        ps.append(pauli_structure(g6str))
        prog.step("features")
    prog.done()
    sr_df, ps_df = pd.DataFrame(sr), pd.DataFrame(ps)
    # Drop any stale persisted copies of these columns so freshly computed values win
    # and the concat produces no duplicate names.
    dup = (set(sr_df.columns) | set(ps_df.columns)) & set(df.columns)
    df = pd.concat([df.drop(columns=list(dup)).reset_index(drop=True), sr_df, ps_df],
                   axis=1)

    classical = _bm.FEATURES
    sr_cols = list(sr[0].keys())
    ps_cols = list(ps[0].keys())
    print(f"  classical {len(classical)}, spectral-rich {len(sr_cols)}, "
          f"pauli-structure {len(ps_cols)}")

    sets = {
        "baseline (classical)": classical,
        "+spectral-rich": classical + sr_cols,
        "+pauli-structure": classical + ps_cols,
        "+both": classical + sr_cols + ps_cols,
    }

    rows = []
    print(f"\n{'feature set':<24}{'best model':<13}{'MCC':>8}{'savings':>10}")
    for label, cols in sets.items():
        best, (mcc, sav), per_model = _score_set_features(df, cols, args.seeds)
        print(f"  {label:<22}{best:<13}{mcc:>8.3f}{sav:>10.3f}")
        for model, (m, s) in per_model.items():
            rows.append({"feature_set": label, "model": model, "mcc": m,
                         "savings": s})

    out = OUT / "enrich_features_mckay.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nReference: classical 0.150, +spectrum(prev) 0.242. spectral-rich is pure")
    print("graph features. pauli-structure is derived from the decomposition (report")
    print("separately: it is the mechanism, so treat any lift there as a cheap-predictor")
    print("result, not a pure-topology one).")
    return 0


# allmetrics mode
def run_allmetrics(args) -> int:
    df = _bm.load(args.dataset)
    df = df[df.n_vertices == 8].reset_index(drop=True)

    # Decomposition features are already persisted in graph_features.csv, so load()
    # returns them, read straight off the frame rather than recomputing.
    classical = list(_bm.FEATURES)
    # Only the two raw counts. Derived diff/ratio and Hamming-weight stats excluded.
    decomp = ["pauli_num_terms", "match_num_matchings"]
    print(f"{args.dataset}: {len(df)} graphs. decomposition features: {decomp}")
    sets = {"classical": classical,
            "+decomp": classical + decomp,
            "decomp-only": decomp}

    rows = []
    fit = Progress(total=len(sets) * len(_bm.learned_models()) * len(SEEDS),
                   unit="fit")
    for fs, cols in sets.items():
        for name, (est, needs_scale) in _bm.learned_models().items():
            acc = {m: [] for m in METRICS}
            for seed in SEEDS:
                tr, _v, te = _bm.split(df, seed)
                Xtr, Xte = tr[cols], te[cols]
                if needs_scale:
                    sc = StandardScaler().fit(Xtr)
                    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
                else:
                    # numpy arrays for consistent XGBoost/LightGBM input type
                    Xtr, Xte = Xtr.to_numpy(), Xte.to_numpy()
                clf = clone(est).fit(Xtr, tr.y)
                pred = clf.predict(Xte)
                s = (clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba")
                     else clf.decision_function(Xte)
                     if hasattr(clf, "decision_function") else None)
                m = _bm.metrics(te.y.to_numpy(), pred, te.delta_cx.to_numpy(), s)
                for k in METRICS:
                    acc[k].append(m[k])
                fit.step(f"{fs}/{name} seed={seed}")
            row = {"feature_set": fs, "model": name}
            for k in METRICS:
                row[k] = float(np.nanmean(acc[k]))
                row[f"{k}_std"] = float(np.nanstd(acc[k]))
            rows.append(row)
    fit.done()

    out = pd.DataFrame(rows)
    out.to_csv(ALLMETRICS_OUT, index=False)
    print(f"\nwrote {ALLMETRICS_OUT}")
    for fs in sets:
        sub = out[out.feature_set == fs].sort_values("mcc", ascending=False)
        best = sub.iloc[0]
        print(f"  {fs:12} best {best.model:12} MCC {best.mcc:.3f} "
              f"savings {best.savings:+.3f}")
    return 0


# spectral mode
def spectral_features(g6_path: Path, n_expected: "int | None") -> pd.DataFrame:
    """One row per graph of spectral summary stats, plus the full sorted spectrum when
    a single vertex count is in play.

    Rows follow g6 order (same as graph_features.csv) for a positional join. Summaries
    are size-agnostic. The per-eigenvalue columns are added only when n_expected is a
    single size, since they would misalign across sizes (pass None to omit them).
    """
    rows = []
    for line in g6_path.read_text().split():
        G = nx.from_graph6_bytes(line.encode())
        n = G.number_of_nodes()
        A = nx.to_numpy_array(G)
        L = np.diag(A.sum(axis=1)) - A

        lap = np.sort(eigvalsh(L))          # ascending, lap[0] ~ 0 for connected
        adj = np.sort(eigvalsh(A))

        # Size-agnostic summaries, normalized where a raw value would scale with n.
        fiedler = lap[1] if len(lap) > 1 else 0.0
        lap_mode2 = lap[2] if len(lap) > 2 else 0.0

        row = {
            "lap_fiedler": fiedler,
            "lap_mode2": lap_mode2,
            "lap_max": lap[-1],
            "lap_mean": lap.mean(),
            "lap_std": lap.std(),
            "lap_energy_norm": np.abs(lap).sum() / n,     # per-vertex, size-comparable
            "adj_max": adj[-1],
            "adj_min": adj[0],
            "adj_gap": adj[-1] - adj[-2] if len(adj) > 1 else 0.0,
            "adj_energy_norm": np.abs(adj).sum() / n,     # graph energy per vertex
            "adj_spread": adj[-1] - adj[0],
        }
        # Full sorted spectra as fixed-width columns, only when a single size is used.
        if n_expected is not None and n == n_expected:
            for i in range(n):
                row[f"lap_ev{i}"] = lap[i]
                row[f"adj_ev{i}"] = adj[i]
        rows.append(row)
    return pd.DataFrame(rows)


def _score_set_spectral(df: pd.DataFrame, feat_cols: list, seeds: list) -> dict:
    """Model comparison (twelve classifiers) on a column set.
    return {model: (mcc, savings)} over seeds."""
    from sklearn.preprocessing import StandardScaler

    out = {}
    for name, (est, needs_scale) in _bm.learned_models().items():
        mccs, savs = [], []
        for seed in seeds:
            tr, _v, te = _bm.split(df, seed)
            if needs_scale:
                sc = StandardScaler().fit(tr[feat_cols])
                Xtr, Xte = sc.transform(tr[feat_cols]), sc.transform(te[feat_cols])
            else:
                Xtr, Xte = tr[feat_cols], te[feat_cols]
            clf = clone(est).fit(Xtr, tr.y)
            pred = clf.predict(Xte)
            if hasattr(clf, "predict_proba"):
                s = clf.predict_proba(Xte)[:, 1]
            elif hasattr(clf, "decision_function"):
                s = clf.decision_function(Xte)
            else:
                s = None
            m = _bm.metrics(te.y.to_numpy(), pred, te.delta_cx.to_numpy(), s)
            mccs.append(m["mcc"])
            savs.append(m["savings"])
        out[name] = (float(np.mean(mccs)), float(np.nanmean(savs)))
    return out


def run_spectral(args) -> int:
    df = _bm.load(args.dataset)
    if args.all_sizes:
        n_mode = None
        print(f"{args.dataset}: {len(df)} graphs across "
              f"{df.n_vertices.nunique()} sizes, {int(df.y.sum())} matching-wins "
              f"({100 * df.y.mean():.2f}%), size-agnostic spectral summaries only")
    else:
        # Single vertex count so the sorted spectrum lines up as fixed columns.
        n_mode = int(df.n_vertices.mode().iloc[0])
        df = df[df.n_vertices == n_mode].reset_index(drop=True)
        print(f"{args.dataset}: {len(df)} graphs at n={n_mode}, "
              f"{int(df.y.sum())} matching-wins ({100 * df.y.mean():.2f}%)")

    g6 = _ROOT / "data" / args.dataset / "g6" / "all.g6"
    print("computing spectral features (Laplacian + adjacency eigenvalues)...")
    # all.g6 and graph_features.csv share row order, so tagging by graph_id then merging
    # aligns the recomputed spectra to the labeled/size-filtered subset.
    spec_all = spectral_features(g6, n_mode)
    feat_full = pd.read_csv(_ROOT / "data" / args.dataset / "features"
                            / "graph_features.csv").reset_index(drop=True)
    spec_all["graph_id"] = feat_full["graph_id"].to_numpy()[: len(spec_all)]
    df = df.merge(spec_all, on="graph_id", how="inner")

    classical = _bm.FEATURES
    spectral = [c for c in spec_all.columns if c != "graph_id"]
    print(f"  classical features: {len(classical)}, spectral features: {len(spectral)}")

    sets = {
        "baseline (classical)": classical,
        "+spectrum": classical + spectral,
        "spectrum only": spectral,
    }
    prog = Progress(total=len(sets) * len(_bm.learned_models()) * len(args.seeds),
                    unit="fit")

    results = {}
    for label, cols in sets.items():
        results[label] = _score_set_spectral(df, cols, args.seeds)
        for _ in _bm.learned_models():
            for _ in args.seeds:
                prog.step(label)
    prog.done()

    # Best MCC + savings per feature set, plus the full per-model table.
    rows = []
    print(f"\n{'feature set':<22}{'best model':<14}{'best MCC':>10}{'its savings':>14}")
    for label, res in results.items():
        best = max(res, key=lambda k: res[k][0])
        mcc, sav = res[best]
        print(f"  {label:<20}{best:<14}{mcc:>10.3f}{sav:>14.3f}")
        for model, (m, s) in res.items():
            rows.append({"feature_set": label, "model": model,
                         "mcc": m, "savings": s})

    out = OUT / f"spectral_features_{args.dataset}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nReading: if +spectrum stays at the baseline MCC (~0.15 on mckay) with")
    print("negative savings, the target is not learnable from the spectrum either, and")
    print("the negative result is representation independent.")
    return 0


# CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Feature-set evaluation experiments (features / allmetrics / spectral).")
    p.add_argument("--mode", choices=["features", "allmetrics", "spectral"],
                   default="allmetrics",
                   help="which experiment to run (default: allmetrics)")
    p.add_argument("--dataset", default="mckay",
                   help="dataset to test (default: mckay)")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    # features mode only
    p.add_argument("--sample", type=int, default=0,
                   help="[features mode] use only N graphs (quick check, 0 = all)")
    # spectral mode only
    p.add_argument("--all-sizes", action="store_true",
                   help="[spectral mode] keep every vertex count and use only "
                        "size-agnostic spectral summaries (for multi-size datasets like "
                        "er/structured). Default restricts to the dominant size and adds "
                        "the full sorted spectrum.")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.mode == "features":
        return run_features(args)
    if args.mode == "allmetrics":
        return run_allmetrics(args)
    return run_spectral(args)


if __name__ == "__main__":
    sys.exit(main())
