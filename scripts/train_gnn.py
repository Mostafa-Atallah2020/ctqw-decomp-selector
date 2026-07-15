#!/usr/bin/env python3
"""Command-line interface for the graph neural network baseline.

Outputs are written to results/gnn_model/:

    metrics.csv               the same twelve metrics as the feature-based ladder,
                              per corpus, for direct comparison
    metrics_by_size.csv       MCC within each vertex count, so that a pooled score
                              cannot conceal a size-based shortcut
    training_history.csv      per-epoch train/validation log-loss
    test_predictions.csv      per-graph ground truth and P(matching)
    *.pdf                     training curve, ROC/PR, confusion, calibration

Four of the six logistic-model diagnostic plots are reused unchanged. Coefficient and
decision-boundary plots are not produced, since a GIN has neither per-feature weights
nor an explicit feature space; the absence reflects the interpretability traded for
direct structural access.

Model and training logic are defined in src/utils/model/gnn.py.

    python scripts/train_gnn.py
    python scripts/train_gnn.py --corpus structured
    python scripts/train_gnn.py --corpus er structured --epochs 30 --seeds 0 1
    python scripts/train_gnn.py --no-plots
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.analysis.corpus_analysis import DEFAULT_DATA  # noqa: E402
from utils.progress import fmt_duration  # noqa: E402

log = logging.getLogger("gnn")


def _print_report(report: dict) -> None:
    """Print all twelve metrics in the column order of
    results/evaluation/model_comparison.csv, so that the GIN row is directly
    comparable to the twelve-model ladder."""
    print(f"\nGIN, mean over {len(report['seeds'])} seeds "
          f"({report['epochs']} epochs each). Same twelve metrics as the ladder:\n")

    cols = [("accuracy", "acc"), ("precision", "prec"), ("recall", "recall"),
            ("specificity", "spec"), ("npv", "npv"), ("f1", "F1"),
            ("mcc", "MCC"), ("cohen_kappa", "kappa"), ("roc_auc", "ROC"),
            ("pr_auc", "PR-AUC"), ("log_loss", "logloss")]

    print("  " + f"{'corpus':<12}" + "".join(f"{lab:>8}" for _, lab in cols)
          + f"{'savings':>13}")
    for r in report["summary"]:
        line = f"  {r['corpus']:<12}"
        for key, _ in cols:
            v = r[f"{key}_mean"]
            line += "     n/a" if v != v else f"{v:>8.3f}"   # v != v catches NaN
        sav = r["savings_mean"]
        line += (f"{sav:>13.3f}" if abs(sav) < 1e3 else f"{sav:>13.2e}")
        print(line)

    print(f"\n  MCC across seeds: " + ", ".join(
        f"{r['corpus']} {r['mcc_mean']:.3f} +/- {r['mcc_std']:.3f}"
        for r in report["summary"]))

    print("\nwithin-size MCC (a pooled score can mask a size-based shortcut):")
    for r in report["by_size"]:
        val = ("single class" if r["single_class"]
               else f"MCC {r['mcc']:>7.3f}")
        print(f"  {r['corpus']:<12} n={r['n_vertices']:>4}  {val:<16}"
              f"({r['n_graphs']} graphs, {r['matching_wins']} matching-wins)")

    print("\nAssess these results by CX savings rather than MCC. Under severe imbalance")
    print("a model can raise MCC by predicting matching more often while reducing")
    print("realized savings: on ER the constant-Pauli baseline attains MCC 0 with 0")
    print("savings, and every method exceeding it on MCC has negative savings.")


def main(argv=None) -> int:
    _t0 = time.perf_counter()
    p = argparse.ArgumentParser(
        description="Benchmark a graph neural network against the feature-based "
                    "models.")
    p.add_argument("--corpus", nargs="+", default=["er", "structured"],
                   choices=["er", "structured", "census8"],
                   help="corpora to evaluate (default: er structured). census8 is the "
                        "exhaustive set of all 11,117 connected 8-vertex graphs")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help=f"data root (default: {DEFAULT_DATA})")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: results/gnn_model)")
    p.add_argument("--epochs", type=int, default=60,
                   help="training epochs per seed (default: 60)")
    p.add_argument("--nmax", type=int, default=256,
                   help="largest graph size to include (default: 256, the largest "
                        "labeled size)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="random seeds (default: 0 1 2 3 4)")
    p.add_argument("--no-plots", action="store_true",
                   help="skip building the PDF plots")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    for corpus in args.corpus:
        if not (args.data / corpus / "g6" / "all.g6").exists():
            log.error("missing graphs for '%s'. Run `scripts/generate_graphs.py "
                      "--corpus %s` first.", corpus, corpus)
            return 2
        if not (args.data / corpus / "labels" / "labels.csv").exists():
            log.error("missing labels for '%s'. Run `scripts/label_graphs.py "
                      "--corpus %s` first.", corpus, corpus)
            return 2

    # Imported only to check they are installed; the real use is in utils.model.gnn.
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401  # type: ignore[import-not-found]
    except ImportError as e:
        log.error("missing dependency: %s.", e)
        log.error("The GNN needs PyTorch, which is not installed by default:")
        log.error("  pip install torch --index-url "
                  "https://download.pytorch.org/whl/cpu")
        log.error("  pip install torch_geometric")
        return 2

    from utils.model.gnn import DEFAULT_OUT, plot_all, run  # noqa: E402

    # Matplotlib logs every glyph it subsets into a PDF at INFO level, burying our
    # output under font internals. Only surface its warnings.
    for noisy in ("matplotlib", "matplotlib.font_manager", "fontTools",
                  "fontTools.subset", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    report = run(args.data, args.out or DEFAULT_OUT, corpora=args.corpus,
                 seeds=args.seeds, epochs=args.epochs, nmax=args.nmax)
    _print_report(report)

    if not args.no_plots:
        plot_all(args.out or DEFAULT_OUT)

    print(f"\nCSVs + plots -> {report['out_dir']}")
    print(f"  done in {fmt_duration(time.perf_counter() - _t0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
