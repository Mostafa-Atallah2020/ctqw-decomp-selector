#!/usr/bin/env python3
"""Train the logistic-regression decomposition selector (CLI).

Fits logistic regression per corpus (er, structured, hybrid) with a leakage-safe
train/val/test split stratified by (size, class), and writes to results/model/:

    metrics.csv            held-out TEST metrics per corpus
    training_history.csv   per-iteration train/val metrics
    test_predictions.csv   per-graph test predictions (y_true, proba, n_vertices)
    coefficients.csv       learned per-corpus feature weights
    *.pdf                  training curves, ROC/PR, confusion, calibration,
                           coefficients, feature-vs-delta, decision boundaries

Logic lives in src/utils/model/model.py and src/utils/model/plots.py.

    python scripts/train_model.py
    python scripts/train_model.py --data F:/data --out F:/out --no-plots
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.model.model import DEFAULT_OUT, run  # noqa: E402
from utils.analysis.corpus_analysis import DEFAULT_DATA  # noqa: E402

log = logging.getLogger("model")


def _print_report(report: dict) -> None:
    print(f"\n{report['n_features']} features, target = {report['target']}")
    print("\nheld-out TEST metrics (logistic regression), per corpus:")
    show = ["accuracy", "f1", "mcc", "roc_auc", "pr_auc", "savings_captured"]
    hdr = "  {:<12}".format("corpus") + "".join(f"{m[:9]:>11}" for m in show)
    print(hdr)
    for r in report["test_metrics"]:
        line = "  {:<12}".format(r["split"])
        line += "".join(f"{str(r.get(m, '')):>11}" for m in show)
        print(line)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Train the logistic-regression decomposition selector.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help=f"data root (default: {DEFAULT_DATA})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"model output dir (default: {DEFAULT_OUT})")
    p.add_argument("--no-plots", action="store_true",
                   help="skip building the PDF plots")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if not (args.data / "hybrid" / "labels" / "labels.csv").exists():
        log.error("missing hybrid corpus. Run `scripts/build_hybrid.py` first.")
        return 2

    try:
        import pandas, sklearn  # noqa: F401
    except ImportError as e:
        log.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    report = run(args.data, args.out)
    _print_report(report)

    if not args.no_plots:
        from utils.model.plots import plot_all
        plot_all(args.out)

    print(f"\nmetrics + history + plots -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
