#!/usr/bin/env python3
"""Dataset build and analysis pipeline (CLI entry point).

Six stages selected with --stage. Each keeps its own flags.

    generate   synthetic g6 dataset
    label      ground-truth decomposition-cost labels
    features   per-graph topology features
    persist    compute/refresh enriched (spectral + Pauli) features into graph_features.csv
    balanced   within-n class-balanced training set
    analyze    comparative analysis of the er and structured datasets

Examples:

    # generate: ER / structured g6 datasets
    python scripts/dataset.py --stage generate --dataset er
    python scripts/dataset.py --stage generate --dataset structured
    python scripts/dataset.py --stage generate --dataset er --limit 100
    python scripts/dataset.py --stage generate --dataset structured -n 100 \
        --vertices 8 16 32 64 128

    # label: compute matching/Pauli cost labels (slow, resumable)
    python scripts/dataset.py --stage label --dataset er
    python scripts/dataset.py --stage label --dataset structured
    python scripts/dataset.py --stage label --dataset er --limit 50
    python scripts/dataset.py --stage label --in F:/data/all.g6 --out F:/data/labels

    # features: extract graph-topology features
    python scripts/dataset.py --stage features --dataset er
    python scripts/dataset.py --stage features --dataset structured
    python scripts/dataset.py --stage features --dataset er --limit 100
    python scripts/dataset.py --stage features --in F:/data/all.g6 --out F:/data/feat

    # persist: compute/refresh enriched spectral + Pauli features
    python scripts/dataset.py --stage persist --dataset mckay
    python scripts/dataset.py --stage persist --dataset er --nmax 256
    python scripts/dataset.py --stage persist --dataset mckay --mode full

    # balanced: build within-n class-balanced set from er + structured
    python scripts/dataset.py --stage balanced
    python scripts/dataset.py --stage balanced --seed 1 --out F:/data/balanced

    # analyze: comparative analysis of the er and structured datasets
    python scripts/dataset.py --stage analyze
    python scripts/dataset.py --stage analyze --data F:/data --plots F:/out
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh

# Make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from progress import Progress, fmt_duration  # noqa: E402
from dataset import generate as _generate  # noqa: E402
from dataset import label as _label  # noqa: E402
from dataset import features as _features  # noqa: E402
from dataset.features import spectral_rich, pauli_structure  # noqa: E402
from dataset.balanced import DEFAULT_DATA, DEFAULT_OUT, build  # noqa: E402
from analysis.dataset_analysis import (  # noqa: E402
    DATASET_NAMES as _DATASET_NAMES,
    DEFAULT_PLOTS as _DEFAULT_PLOTS,
    run as _analyze_run,
)

_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA = _REPO_ROOT / "data"

log = logging.getLogger("dataset_build")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def run_generate(args) -> int:
    logg = logging.getLogger("generate")

    try:
        import networkx  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        logg.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    bad = [n for n in args.vertices if not _generate.is_power_of_two(n)]
    if bad:
        logg.error("vertex counts must be powers of 2, offending: %s", bad)
        return 2

    default_out = (_generate.DEFAULT_ER_OUT if args.dataset == "er"
                   else _generate.DEFAULT_STRUCTURED_OUT)
    out_dir = args.out or default_out
    out_g6 = out_dir / "all.g6"
    manifest_path = out_dir / "manifest.json"

    stats = _generate.Stats()
    if args.dataset == "er":
        require_connected = not args.allow_disconnected
        params = {
            "dataset": "er",
            "vertices": args.vertices,
            "probabilities": args.probability,
            "n_graphs_per_combo": args.n_graphs,
            "base_seed": args.seed,
            "require_connected": require_connected,
            "limit": args.limit,
        }
        _generate.generate_er_dataset(
            out_g6, args.vertices, args.probability, args.n_graphs,
            args.seed, stats, require_connected=require_connected,
            limit=args.limit)
        source = "erdos_renyi_synthetic"
    else:
        params = {
            "dataset": "structured",
            "vertices": args.vertices,
            "n_graphs_per_size": args.n_graphs,
            "base_seed": args.seed,
            "limit": args.limit,
        }
        _generate.generate_structured_dataset(
            out_g6, args.vertices, args.n_graphs,
            args.seed, stats, limit=args.limit)
        source = "structured_counting_path"

    _generate.write_manifest(stats, out_g6, manifest_path, params, source)

    print("\nDone.")
    print(f"  dataset:              {args.dataset}")
    print(f"  generated:           {stats.generated}")
    print(f"  skipped_duplicate:   {stats.skipped_duplicate}")
    print(f"  skipped_disconnected:{stats.skipped_disconnected}")
    print(f"  size histogram:      {dict(sorted(stats.size_histogram.items()))}")
    if stats.short_runs:
        print(f"  short runs:          {stats.short_runs}")
    print(f"  g6 output:           {out_g6}")
    return 0


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------
def run_label(args) -> int:
    logg = logging.getLogger("label")

    # Qiskit's transpiler logs every pass at INFO, which would flood a long run
    # and bury the progress line. Quiet it unless -v was requested.
    if not args.verbose:
        logging.getLogger("qiskit").setLevel(logging.WARNING)

    g6 = args.g6 or (_DATA / args.dataset / "g6" / "all.g6")
    out_dir = args.out or (_DATA / args.dataset / "labels")

    if not g6.exists():
        logg.error("g6 dataset not found: %s. Run "
                   "`scripts/dataset.py --stage generate --dataset %s` first.",
                   g6, args.dataset)
        return 2

    try:
        import qiskit  # noqa: F401
        import ctqw_matching_decomp  # noqa: F401
    except ImportError as e:
        logg.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    labels_csv = out_dir / "labels.csv"
    manifest_path = out_dir / "labels_manifest.json"

    stats = _label.Stats()
    _label.label_dataset(g6, labels_csv, stats, limit=args.limit)
    _label.write_manifest(stats, g6, labels_csv, manifest_path)

    print("\nDone.")
    print(f"  dataset:         {args.dataset}")
    print(f"  labeled (new):  {stats.labeled}")
    print(f"  resumed (skip): {stats.resumed}")
    print(f"  failed:         {stats.failed}")
    print(f"  wall clock:     {_label._fmt_duration(stats.wall_seconds)}")
    print(f"  labels.csv:     {labels_csv}")
    return 0


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def run_features(args) -> int:
    logg = logging.getLogger("features")

    g6 = args.g6 or (_DATA / args.dataset / "g6" / "all.g6")
    out_dir = args.out or (_DATA / args.dataset / "features")

    if not g6.exists():
        logg.error("g6 dataset not found: %s. Run "
                   "`scripts/dataset.py --stage generate --dataset %s` first.",
                   g6, args.dataset)
        return 2

    try:
        import networkx  # noqa: F401
    except ImportError as e:
        logg.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    graph_csv = out_dir / "graph_features.csv"
    manifest_path = out_dir / "features_manifest.json"

    stats = _features.Stats()
    _features.extract_features(g6, graph_csv, stats, limit=args.limit)
    _features.write_manifest(stats, g6, graph_csv, manifest_path)

    print("\nDone.")
    print(f"  dataset:               {args.dataset}")
    print(f"  parsed:               {stats.parsed}")
    print(f"  skipped_parse_error:  {stats.skipped_parse_error}")
    print(f"  skipped_disconnected: {stats.skipped_disconnected}")
    print(f"  duplicate_graph:      {stats.duplicate_graph}")
    print(f"  graph_features.csv:   {graph_csv}")
    return 0


# ---------------------------------------------------------------------------
# balanced
# ---------------------------------------------------------------------------
def run_balanced(args) -> int:
    logg = logging.getLogger("balanced")
    _t0 = time.perf_counter()

    for dataset in ("er", "structured"):
        if not (args.data / dataset / "labels" / "labels.csv").exists():
            logg.error("missing labels for '%s'. Label it first.", dataset)
            return 2

    m = build(args.data, args.out, seed=args.seed)

    print("\nDone.")
    print(f"  labeled:       {m['total']}")
    print(f"  matching wins: {m['matching_wins']}")
    print(f"  pauli wins:    {m['pauli_wins']}")
    print(f"  source mix:    {m['source_mix']}")
    print("  per-n (take/class, total):")
    for n, s in m["per_n"].items():
        print(f"    n={n:>4}: {s['take_per_class']}/class  = {s['total']}")

    unl = m.get("unlabeled_carried", {})
    if unl.get("total"):
        per_size = ", ".join(f"n={n}: {c}" for n, c in unl["per_size"].items())
        print(f"  unlabeled:     {unl['total']} carried into features only "
              f"({per_size}, {unl['per_dataset']})")

    print(f"  -> {args.out}")
    print(f"  done in {fmt_duration(time.perf_counter() - _t0)}")
    return 0


# ---------------------------------------------------------------------------
# persist
# ---------------------------------------------------------------------------
# persisted column -> key returned by pauli_structure (used by the refresh path)
COLS = {
    "pauli_num_terms": "pauli_num_terms",
    "match_num_matchings": "num_matchings",
    "pauli_match_ratio": "term_ratio",
    "pauli_match_diff": "term_diff",
    "pauli_wt_mean": "pauli_wt_mean",
    "pauli_wt_max": "pauli_wt_max",
    "pauli_wt_std": "pauli_wt_std",
}
#: Marks rows written by the refresh path, so a restart resumes rather than redoing.
STAMP = "decomp_refreshed"


def _basic_spectrum(G: "nx.Graph") -> dict:
    """Size-agnostic Laplacian/adjacency spectral summaries, prefixed spec_."""
    A = nx.to_numpy_array(G)
    n = G.number_of_nodes()
    L = np.diag(A.sum(axis=1)) - A
    lap = np.sort(eigvalsh(L))
    adj = np.sort(eigvalsh(A))
    return {
        "spec_lap_fiedler": lap[1] if len(lap) > 1 else 0.0,
        "spec_lap_mode2": lap[2] if len(lap) > 2 else 0.0,
        "spec_lap_max": lap[-1],
        "spec_lap_mean": lap.mean(),
        "spec_lap_std": lap.std(),
        "spec_lap_energy_norm": np.abs(lap).sum() / n,
        "spec_adj_max": adj[-1],
        "spec_adj_min": adj[0],
        "spec_adj_gap": adj[-1] - adj[-2] if len(adj) > 1 else 0.0,
        "spec_adj_energy_norm": np.abs(adj).sum() / n,
        "spec_adj_spread": adj[-1] - adj[0],
    }


def _pauli_row(g6str: str) -> dict:
    """The decomposition-internal features (prefixed to flag leakage). Expensive on
    large dense graphs."""
    ps = pauli_structure(g6str)
    return {
        "pauli_num_terms": ps["pauli_num_terms"],
        "match_num_matchings": ps["num_matchings"],
        "pauli_match_ratio": ps["term_ratio"],
        "pauli_match_diff": ps["term_diff"],
        "pauli_wt_mean": ps["pauli_wt_mean"],
        "pauli_wt_max": ps["pauli_wt_max"],
        "pauli_wt_std": ps["pauli_wt_std"],
    }


def run_full(args) -> int:
    cdir = _ROOT / "data" / args.dataset
    feat_path = cdir / "features" / "graph_features.csv"
    g6 = (cdir / "g6" / "all.g6").read_text().split()
    df = pd.read_csv(feat_path).reset_index(drop=True)

    # Refuse to run on a truncated features file (row count must match the g6 dataset).
    if len(df) != len(g6):
        raise SystemExit(
            f"{feat_path} has {len(df)} rows but the dataset has {len(g6)} graphs. "
            f"The file is truncated. Restore graph_features.classical.bak.csv (or the "
            f"committed version) before re-running.")

    g6_by_id = dict(zip(df["graph_id"], g6[: len(df)]))

    # Only compute rows up to nmax. Larger rows keep NaN in the new columns.
    todo = df[df.n_vertices <= args.nmax]

    # Resume support: skip rows a prior run already filled.
    done_ids = set()
    if "pauli_num_terms" in df.columns and not args.no_pauli:
        done_ids = set(df.loc[df["pauli_num_terms"].notna(), "graph_id"])
    print(f"{args.dataset}: {len(df)} graphs, enriching {len(todo)} at n<={args.nmax}")
    if not args.no_pauli:
        n_pauli = int((todo.n_vertices <= args.pauli_nmax).sum())
        print(f"  Pauli/matching for {n_pauli} graphs at n<={args.pauli_nmax} "
              f"(expensive). {len(done_ids)} already done, "
              f"checkpointing every {args.checkpoint}")

    # Accumulate in a dict keyed by graph_id and flush every `checkpoint` graphs.
    def _flush(results):
        # Left-join the added columns onto the full df (missing rows -> NaN). Returns
        # the added column names.
        add = pd.DataFrame.from_dict(results, orient="index")
        add.index.name = "graph_id"
        add = add.reset_index()
        base_cols = [c for c in df.columns
                     if c not in add.columns or c == "graph_id"]
        out = df[base_cols].merge(add, on="graph_id", how="left")
        assert len(out) == len(df), f"row count changed: {len(out)} != {len(df)}"
        out.to_csv(feat_path, index=False)
        return [c for c in add.columns if c != "graph_id"]

    # Seed `results` with every row already carrying any added feature, so _flush
    # (which rebuilds columns solely from `results`) does not wipe cached rows to NaN.
    results = {}
    added_cols = [c for c in df.columns if c.startswith(("spec_", "pauli_", "match_"))]
    if added_cols:
        has_any = df[added_cols].notna().any(axis=1)
        prev = df.loc[has_any].set_index("graph_id")[added_cols]
        results = {gid: row.dropna().to_dict() for gid, row in prev.iterrows()}
    # Rows with spectral features already present are skipped on resume.
    spec_done = set()
    if "spec_lap_max" in df.columns:
        spec_done = set(df.loc[df["spec_lap_max"].notna(), "graph_id"])

    prog = Progress(total=len(todo), unit="graph")
    since_flush = 0
    for gid, nv in zip(todo["graph_id"].tolist(), todo["n_vertices"].tolist()):
        spec_needed = not args.no_spectral
        pauli_needed = (not args.no_pauli) and nv <= args.pauli_nmax
        # A row is "already done" when every feature this run would compute is present.
        spec_ok = (not spec_needed) or gid in spec_done
        pauli_ok = (not pauli_needed) or gid in done_ids
        if spec_ok and pauli_ok:
            prog.step(f"{args.dataset} n={int(nv)} (cached)", key=args.dataset)
            continue

        g6str = g6_by_id[gid]
        G = nx.from_graph6_bytes(g6str.encode())
        # Keep cached features, add only the missing groups.
        row = dict(results.get(gid, {}))
        if spec_needed and gid not in spec_done:
            row.update(_basic_spectrum(G))
            row.update({f"spec_{k}": v for k, v in spectral_rich(G).items()})
        if pauli_needed and gid not in done_ids:
            row.update(_pauli_row(g6str))
        results[gid] = row

        since_flush += 1
        if since_flush >= args.checkpoint:
            _flush(results)                       # real-time save
            since_flush = 0
        prog.step(f"{args.dataset} n={int(nv)}", key=args.dataset)
    prog.done()

    # Final flush (also covers the all-cached case).
    added = _flush(results)
    topo = [c for c in added if c.startswith("spec_")]
    leaky = [c for c in added if c.startswith(("pauli_", "match_"))]
    total_cols = len(pd.read_csv(feat_path, nrows=0).columns)
    print(f"\nwrote {feat_path}")
    print(f"  +{len(topo)} spec_* topology features (pure), "
          f"+{len(leaky)} pauli_/match_ features (LEAKY)")
    print(f"  total columns now {total_cols}, "
          f"rows with NaN new cols (n>{args.nmax}): "
          f"{int((df.n_vertices > args.nmax).sum())}")
    return 0


def run_refresh(args) -> int:
    cdir = _ROOT / "data" / args.dataset
    feat_path = cdir / "features" / "graph_features.csv"
    g6 = (cdir / "g6" / "all.g6").read_text().split()
    df = pd.read_csv(feat_path).reset_index(drop=True)

    # Guard: the features file must line up one-to-one with the graph dataset.
    if len(df) != len(g6):
        raise SystemExit(
            f"{feat_path} has {len(df)} rows but the dataset has {len(g6)} graphs. "
            f"Refusing to run on a mismatched file.")

    if STAMP not in df.columns:
        df[STAMP] = False
    df[STAMP] = df[STAMP].fillna(False).astype(bool)

    todo = df.index[(df.n_vertices <= args.nmax) & (~df[STAMP])].tolist()
    done_already = int(df[STAMP].sum())
    print(f"{args.dataset}: {len(df)} graphs, {done_already} already refreshed, "
          f"{len(todo)} to compute at n<={args.nmax}")
    if not todo:
        print("nothing to do")
        return 0

    prog = Progress(total=len(todo), unit="graph")
    t0 = time.perf_counter()
    since = 0
    for i in todo:
        feats = pauli_structure(g6[i])
        for col, key in COLS.items():
            df.at[i, col] = float(feats[key])
        df.at[i, STAMP] = True
        since += 1
        if since >= args.checkpoint:
            df.to_csv(feat_path, index=False)
            since = 0
        prog.step(f"{args.dataset} n={int(df.at[i, 'n_vertices'])}", key=args.dataset)
    prog.done()

    df.to_csv(feat_path, index=False)
    dt = time.perf_counter() - t0
    print(f"\nwrote {feat_path}")
    print(f"  {len(todo)} graphs in {dt:.1f}s ({dt / max(len(todo), 1):.3f}s per graph)")
    filled = int(df[list(COLS)].notna().all(axis=1).sum())
    print(f"  rows with all {len(COLS)} decomposition features: {filled}/{len(df)}")
    return 0


def run_persist(args) -> int:
    # Per-mode checkpoint default: full=50,
    # refresh=200.
    if args.checkpoint is None:
        args.checkpoint = 50 if args.mode == "full" else 200

    if args.mode == "full":
        return run_full(args)
    return run_refresh(args)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def _analyze_print_report(report: dict) -> None:
    for dataset in _DATASET_NAMES:
        r = report[dataset]
        lb = r["label_balance"]
        print(f"\n=== {dataset} ({r['n_graphs']} labeled graphs) ===")
        print(f"  matching wins: {lb['matching_wins']} "
              f"({lb['matching_win_frac']*100:.1f}%)")
        print(f"  pauli wins:    {lb['pauli_wins']}")
        print(f"  minority frac: {lb['minority_frac']*100:.1f}%  "
              f"(imbalance {lb['imbalance_ratio']}:1)")
        print(f"  VERDICT: {lb['verdict']}")
        if r["degenerate_features"]:
            print(f"  degenerate features (>=90% one value): "
                  f"{list(r['degenerate_features'])}")
        print("  matching-win rate by n:")
        for n, s in r["win_rate_by_n"].items():
            print(f"    n={n:>4}: {s['matching_win_rate']*100:5.1f}%  "
                  f"(mean delta_cx {s['mean_delta_cx']:+.0f}, {s['count']} graphs)")
    print(f"\nPCA: PC1 {report['pca']['pc1_var']*100:.0f}% + "
          f"PC2 {report['pca']['pc2_var']*100:.0f}% variance")


def run_analyze(args) -> int:
    # Quiet matplotlib/fontTools INFO spam, keep only warnings.
    for noisy in ("matplotlib", "matplotlib.font_manager", "fontTools",
                  "fontTools.subset", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    for dataset in _DATASET_NAMES:
        if not (args.data / dataset / "labels" / "labels.csv").exists():
            hint = ("scripts/dataset.py --stage balanced" if dataset == "balanced"
                    else f"scripts/dataset.py --stage label --dataset {dataset}")
            log.error("missing labels for '%s'. Run `%s` first.", dataset, hint)
            return 2

    try:
        import pandas, seaborn, sklearn  # noqa: F401
    except ImportError as e:
        log.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    report = _analyze_run(args.data, args.plots)
    _analyze_print_report(report)
    print(f"\nplots + balance_report.json -> {args.plots}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_generate_args(sp) -> None:
    sp.add_argument("--dataset", choices=("er", "structured"), required=True,
                    help="which graph family to generate")
    sp.add_argument("--vertices", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128],
                    help="vertex counts, each a power of 2 (default: 8 16 32 64 128)")
    sp.add_argument("-p", "--probability", type=float, nargs="+",
                    default=[round(0.05 * k, 2) for k in range(1, 20)],
                    help="ER edge probabilities (default: 0.05..0.95 step 0.05, "
                         "ignored for --dataset structured)")
    sp.add_argument("-n", "--n-graphs", type=int, default=100,
                    help="unique graphs per cell/size (default: 100)")
    sp.add_argument("--out", type=Path, default=None,
                    help="g6 output dir (default: data/<dataset>/g6)")
    sp.add_argument("--seed", type=int, default=42,
                    help="base random seed (default: 42)")
    sp.add_argument("--allow-disconnected", action="store_true",
                    help="ER only: keep disconnected graphs (default: reject)")
    sp.add_argument("--limit", type=int, default=None,
                    help="write only the first N graphs total (smoke tests)")
    sp.add_argument("-v", "--verbose", action="store_true")


def _add_label_args(sp) -> None:
    sp.add_argument("--dataset", choices=("er", "structured"), default="er",
                    help="dataset to read/write under data/<dataset>/ (default: er)")
    sp.add_argument("--in", dest="g6", type=Path, default=None,
                    help="path to all.g6 (default: data/<dataset>/g6/all.g6)")
    sp.add_argument("--out", type=Path, default=None,
                    help="labels output dir (default: data/<dataset>/labels)")
    sp.add_argument("--limit", type=int, default=None,
                    help="label only N new graphs (for smoke tests)")
    sp.add_argument("-v", "--verbose", action="store_true")


def _add_features_args(sp) -> None:
    sp.add_argument("--dataset", choices=("er", "structured"), default="er",
                    help="dataset to read/write under data/<dataset>/ (default: er)")
    sp.add_argument("--in", dest="g6", type=Path, default=None,
                    help="path to all.g6 (default: data/<dataset>/g6/all.g6)")
    sp.add_argument("--out", type=Path, default=None,
                    help="features output dir (default: data/<dataset>/features)")
    sp.add_argument("--limit", type=int, default=None,
                    help="process only the first N graphs (for smoke tests)")
    sp.add_argument("-v", "--verbose", action="store_true")


def _add_balanced_args(sp) -> None:
    sp.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help=f"data root (default: {DEFAULT_DATA})")
    sp.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"balanced output dir (default: {DEFAULT_OUT})")
    sp.add_argument("--seed", type=int, default=0, help="sampling seed")
    sp.add_argument("-v", "--verbose", action="store_true")


def _add_persist_args(sp) -> None:
    sp.add_argument("--dataset", required=True,
                    help="mckay, er, structured, ...")
    sp.add_argument("--mode", choices=("full", "refresh"), default="refresh",
                    help="full: compute+persist spectral + Pauli features. "
                         "refresh: fast tensorized Pauli recompute only, marker-tracked. "
                         "Default: refresh.")
    # full-mode flags
    sp.add_argument("--pauli-nmax", type=int, default=32,
                    help="[full] largest size for the EXPENSIVE Pauli features "
                         "(default 32, n=128 is ~7.5 s/graph, n=256 ~3 min/graph)")
    sp.add_argument("--no-pauli", action="store_true",
                    help="[full] skip the Pauli/matching features entirely "
                         "(spectral only)")
    sp.add_argument("--no-spectral", action="store_true",
                    help="[full] skip the spectral features entirely "
                         "(decomposition only)")
    # shared flags (defaults differ per mode, resolved in run_persist)
    sp.add_argument("--nmax", type=int, default=256,
                    help="largest size to featurize. In full mode this is the SPECTRAL "
                         "cutoff (default 256), in refresh mode the vertex cutoff "
                         "(default 256).")
    sp.add_argument("--checkpoint", type=int, default=None,
                    help="write the CSV every N graphs so an interrupted run is not lost. "
                         "Default 50 in full mode, 200 in refresh mode.")
    sp.add_argument("-v", "--verbose", action="store_true")


def _add_analyze_args(sp) -> None:
    sp.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help=f"data root (default: {DEFAULT_DATA})")
    sp.add_argument("--plots", type=Path, default=_DEFAULT_PLOTS,
                    help=f"plot output dir (default: {_DEFAULT_PLOTS})")
    sp.add_argument("-v", "--verbose", action="store_true")


_STAGES = {
    "generate": (_add_generate_args, run_generate,
                 "Generate a synthetic g6 dataset (ER or structured)."),
    "label": (_add_label_args, run_label,
              "Compute matching/Pauli decomposition-cost labels."),
    "features": (_add_features_args, run_features,
                 "Extract per-graph topology features from a g6 dataset."),
    "persist": (_add_persist_args, run_persist,
                "Compute/refresh enriched (spectral + Pauli) features into "
                "graph_features.csv."),
    "balanced": (_add_balanced_args, run_balanced,
                 "Build a within-n class-balanced training set."),
    "analyze": (_add_analyze_args, run_analyze,
                "Comparative analysis of the er and structured datasets."),
}


def main(argv=None) -> int:
    # Windows consoles default to cp1252, which can't encode accented chars.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    argv = list(argv) if argv is not None else sys.argv[1:]

    # Route on --stage, then hand remaining flags to a per-stage parser.
    top = argparse.ArgumentParser(
        description="Dataset build and analysis pipeline "
                    "(generate | label | features | persist | balanced | analyze).",
        add_help=False)
    top.add_argument("--stage", choices=tuple(_STAGES), required=True,
                     help="pipeline stage to run")
    stage_ns, rest = top.parse_known_args(argv)

    add_args, run_fn, help_text = _STAGES[stage_ns.stage]
    sp = argparse.ArgumentParser(
        prog=f"dataset.py --stage {stage_ns.stage}",
        description=help_text)
    add_args(sp)
    args = sp.parse_args(rest)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    return run_fn(args)


if __name__ == "__main__":
    sys.exit(main())
