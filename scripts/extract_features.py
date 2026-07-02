#!/usr/bin/env python3
"""Extract graph-topology features from the g6 corpus (CLI entry point).

The extraction logic lives in src/utils/dataset/features_g6.py; this script owns
the command-line interface. Run from the repo root:

    python scripts/extract_features.py                 # full corpus
    python scripts/extract_features.py --limit 100     # smoke test
    python scripts/extract_features.py --in F:/data/all.g6 --out F:/data/feat
"""

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.dataset.features_g6 import (  # noqa: E402
    DEFAULT_G6,
    DEFAULT_OUT,
    Stats,
    extract_features,
    write_manifest,
)

log = logging.getLogger("er.features")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Extract per-graph topology features from a g6 corpus.")
    p.add_argument("--in", dest="g6", type=Path, default=DEFAULT_G6,
                   help=f"path to all.g6 (default: {DEFAULT_G6})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"features output dir (default: {DEFAULT_OUT})")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N graphs (for smoke tests)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if not args.g6.exists():
        log.error("g6 corpus not found: %s. Run scripts/generate_er_graphs.py "
                  "first.", args.g6)
        return 2

    try:
        import networkx  # noqa: F401
    except ImportError as e:
        log.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    graph_csv = args.out / "graph_features.csv"
    manifest_path = args.out / "features_manifest.json"

    stats = Stats()
    extract_features(args.g6, graph_csv, stats, limit=args.limit)
    write_manifest(stats, args.g6, graph_csv, manifest_path)

    print("\nDone.")
    print(f"  parsed:               {stats.parsed}")
    print(f"  skipped_parse_error:  {stats.skipped_parse_error}")
    print(f"  skipped_disconnected: {stats.skipped_disconnected}")
    print(f"  duplicate_graph:      {stats.duplicate_graph}")
    print(f"  graph_features.csv:   {graph_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
