#!/usr/bin/env python3
"""Extract graph-topology features from a g6 corpus (CLI entry point).

Pick the corpus with --corpus; --in / --out override the derived paths.
The extraction logic lives in src/utils/dataset/features_g6.py.

    python scripts/extract_features.py --corpus er                 # ER features
    python scripts/extract_features.py --corpus structured         # structured
    python scripts/extract_features.py --corpus er --limit 100     # smoke test
    python scripts/extract_features.py --in F:/data/all.g6 --out F:/data/feat
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.dataset.features_g6 import (  # noqa: E402
    Stats,
    extract_features,
    write_manifest,
)

log = logging.getLogger("features")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA = _REPO_ROOT / "data"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Extract per-graph topology features from a g6 corpus.")
    p.add_argument("--corpus", choices=("er", "structured"), default="er",
                   help="corpus to read/write under data/<corpus>/ (default: er)")
    p.add_argument("--in", dest="g6", type=Path, default=None,
                   help="path to all.g6 (default: data/<corpus>/g6/all.g6)")
    p.add_argument("--out", type=Path, default=None,
                   help="features output dir (default: data/<corpus>/features)")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N graphs (for smoke tests)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    g6 = args.g6 or (_DATA / args.corpus / "g6" / "all.g6")
    out_dir = args.out or (_DATA / args.corpus / "features")

    if not g6.exists():
        log.error("g6 corpus not found: %s. Run "
                  "`scripts/generate_graphs.py --corpus %s` first.",
                  g6, args.corpus)
        return 2

    try:
        import networkx  # noqa: F401
    except ImportError as e:
        log.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    graph_csv = out_dir / "graph_features.csv"
    manifest_path = out_dir / "features_manifest.json"

    stats = Stats()
    extract_features(g6, graph_csv, stats, limit=args.limit)
    write_manifest(stats, g6, graph_csv, manifest_path)

    print("\nDone.")
    print(f"  corpus:               {args.corpus}")
    print(f"  parsed:               {stats.parsed}")
    print(f"  skipped_parse_error:  {stats.skipped_parse_error}")
    print(f"  skipped_disconnected: {stats.skipped_disconnected}")
    print(f"  duplicate_graph:      {stats.duplicate_graph}")
    print(f"  graph_features.csv:   {graph_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
