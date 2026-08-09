#!/usr/bin/env python3
"""Unified dataset-build pipeline (CLI entry point).

Four stages selected with --stage; each keeps its own flags.

    generate   synthetic g6 dataset
    label      ground-truth decomposition-cost labels
    features   per-graph topology features
    balanced   within-n class-balanced training set

Examples:

    # generate: ER / structured g6 datasets
    python scripts/synthetic_dataset_build.py --stage generate --dataset er
    python scripts/synthetic_dataset_build.py --stage generate --dataset structured
    python scripts/synthetic_dataset_build.py --stage generate --dataset er --limit 100
    python scripts/synthetic_dataset_build.py --stage generate --dataset structured -n 100 \
        --vertices 8 16 32 64 128

    # label: compute matching/Pauli cost labels (slow; resumable)
    python scripts/synthetic_dataset_build.py --stage label --dataset er
    python scripts/synthetic_dataset_build.py --stage label --dataset structured
    python scripts/synthetic_dataset_build.py --stage label --dataset er --limit 50
    python scripts/synthetic_dataset_build.py --stage label --in F:/data/all.g6 --out F:/data/labels

    # features: extract graph-topology features
    python scripts/synthetic_dataset_build.py --stage features --dataset er
    python scripts/synthetic_dataset_build.py --stage features --dataset structured
    python scripts/synthetic_dataset_build.py --stage features --dataset er --limit 100
    python scripts/synthetic_dataset_build.py --stage features --in F:/data/all.g6 --out F:/data/feat

    # balanced: build within-n class-balanced set from er + structured
    python scripts/synthetic_dataset_build.py --stage balanced
    python scripts/synthetic_dataset_build.py --stage balanced --seed 1 --out F:/data/balanced
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.progress import fmt_duration  # noqa: E402
from utils.dataset import generate as _generate  # noqa: E402
from utils.dataset import label as _label  # noqa: E402
from utils.dataset import features as _features  # noqa: E402
from utils.dataset.balanced import DEFAULT_DATA, DEFAULT_OUT, build  # noqa: E402

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
        logg.error("vertex counts must be powers of 2; offending: %s", bad)
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
                   "`scripts/synthetic_dataset_build.py --stage generate --dataset %s` first.",
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
                   "`scripts/synthetic_dataset_build.py --stage generate --dataset %s` first.",
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
              f"({per_size}; {unl['per_dataset']})")

    print(f"  -> {args.out}")
    print(f"  done in {fmt_duration(time.perf_counter() - _t0)}")
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
                    help="ER edge probabilities (default: 0.05..0.95 step 0.05; "
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


_STAGES = {
    "generate": (_add_generate_args, run_generate,
                 "Generate a synthetic g6 dataset (ER or structured)."),
    "label": (_add_label_args, run_label,
              "Compute matching/Pauli decomposition-cost labels."),
    "features": (_add_features_args, run_features,
                 "Extract per-graph topology features from a g6 dataset."),
    "balanced": (_add_balanced_args, run_balanced,
                 "Build a within-n class-balanced balanced training set."),
}


def main(argv=None) -> int:
    # Windows consoles default to cp1252, which can't encode accented chars.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    argv = list(argv) if argv is not None else sys.argv[1:]

    # Route on the required --stage first, then hand the remaining flags to a
    # stage-specific parser so each stage keeps its own flag defaults/semantics
    # (e.g. --dataset is required for generate but defaults to "er" elsewhere).
    top = argparse.ArgumentParser(
        description="Unified dataset-build pipeline "
                    "(generate | label | features | balanced).",
        add_help=False)
    top.add_argument("--stage", choices=tuple(_STAGES), required=True,
                     help="pipeline stage to run")
    stage_ns, rest = top.parse_known_args(argv)

    add_args, run_fn, help_text = _STAGES[stage_ns.stage]
    sp = argparse.ArgumentParser(
        prog=f"synthetic_dataset_build.py --stage {stage_ns.stage}",
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
