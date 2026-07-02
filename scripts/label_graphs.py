#!/usr/bin/env python3
"""Compute ground-truth decomposition-cost labels for the g6 corpus (CLI).

The labeling logic lives in src/utils/dataset/label.py; this script owns the
command-line interface. Labeling is the expensive stage (dominated by Pauli
transpilation at optimization_level=3) but is resumable: re-running skips
graphs already in labels.csv.

    python scripts/label_graphs.py --corpus er              # label ER corpus (slow)
    python scripts/label_graphs.py --corpus structured      # label structured
    python scripts/label_graphs.py --corpus er --limit 50   # smoke test
    python scripts/label_graphs.py --in F:/data/all.g6 --out F:/data/labels
"""

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.dataset.label import (  # noqa: E402
    Stats,
    label_corpus,
    write_manifest,
    _fmt_duration,
)

log = logging.getLogger("label")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA = _REPO_ROOT / "data"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Compute matching/Pauli decomposition-cost labels.")
    p.add_argument("--corpus", choices=("er", "structured"), default="er",
                   help="corpus to read/write under data/<corpus>/ (default: er)")
    p.add_argument("--in", dest="g6", type=Path, default=None,
                   help="path to all.g6 (default: data/<corpus>/g6/all.g6)")
    p.add_argument("--out", type=Path, default=None,
                   help="labels output dir (default: data/<corpus>/labels)")
    p.add_argument("--limit", type=int, default=None,
                   help="label only N new graphs (for smoke tests)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    # Qiskit's transpiler logs every pass at INFO, which would flood a long run
    # and bury the progress line. Quiet it unless -v was requested.
    if not args.verbose:
        logging.getLogger("qiskit").setLevel(logging.WARNING)

    g6 = args.g6 or (_DATA / args.corpus / "g6" / "all.g6")
    out_dir = args.out or (_DATA / args.corpus / "labels")

    if not g6.exists():
        log.error("g6 corpus not found: %s. Run "
                  "`scripts/generate_graphs.py --corpus %s` first.",
                  g6, args.corpus)
        return 2

    try:
        import qiskit  # noqa: F401
        import ctqw_matching_decomp  # noqa: F401
    except ImportError as e:
        log.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    labels_csv = out_dir / "labels.csv"
    manifest_path = out_dir / "labels_manifest.json"

    stats = Stats()
    label_corpus(g6, labels_csv, stats, limit=args.limit)
    write_manifest(stats, g6, labels_csv, manifest_path)

    print("\nDone.")
    print(f"  corpus:         {args.corpus}")
    print(f"  labeled (new):  {stats.labeled}")
    print(f"  resumed (skip): {stats.resumed}")
    print(f"  failed:         {stats.failed}")
    print(f"  wall clock:     {_fmt_duration(stats.wall_seconds)}")
    print(f"  labels.csv:     {labels_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
