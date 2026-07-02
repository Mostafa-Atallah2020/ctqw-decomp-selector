#!/usr/bin/env python3
"""Comparative data analysis of the ER and structured corpora (CLI).

Logic lives in src/utils/analysis/corpus_analysis.py.

    python scripts/analyze_corpora.py
    python scripts/analyze_corpora.py --data F:/data --plots F:/out
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.analysis.corpus_analysis import (  # noqa: E402
    CORPUS_NAMES,
    DEFAULT_DATA,
    DEFAULT_PLOTS,
    run,
)

log = logging.getLogger("analysis")


def _print_report(report: dict) -> None:
    for corpus in CORPUS_NAMES:
        r = report[corpus]
        lb = r["label_balance"]
        print(f"\n=== {corpus} ({r['n_graphs']} labeled graphs) ===")
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Comparative analysis of the er and structured corpora.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help=f"data root (default: {DEFAULT_DATA})")
    p.add_argument("--plots", type=Path, default=DEFAULT_PLOTS,
                   help=f"plot output dir (default: {DEFAULT_PLOTS})")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    for corpus in CORPUS_NAMES:
        if not (args.data / corpus / "labels" / "labels.csv").exists():
            hint = ("scripts/build_hybrid.py" if corpus == "hybrid"
                    else f"scripts/label_graphs.py --corpus {corpus}")
            log.error("missing labels for '%s'. Run `%s` first.", corpus, hint)
            return 2

    try:
        import pandas, seaborn, sklearn  # noqa: F401
    except ImportError as e:
        log.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    report = run(args.data, args.plots)
    _print_report(report)
    print(f"\nplots + balance_report.json -> {args.plots}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
