#!/usr/bin/env python3
"""Generate the synthetic Erdos-Renyi g6 corpus (CLI entry point).

The generation logic lives in src/utils/dataset/generate.py; this script owns
the command-line interface. Run from the repo root:

    python scripts/generate_er_graphs.py                       # default sweep
    python scripts/generate_er_graphs.py --limit 100           # smoke test
    python scripts/generate_er_graphs.py --vertices 8 16 -p 0.1 0.2 -n 200
    python scripts/generate_er_graphs.py --out F:/datasets/er
"""

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.dataset.generate import (  # noqa: E402
    DEFAULT_OUT,
    Stats,
    generate_corpus,
    write_manifest,
    is_power_of_two,
)

log = logging.getLogger("er.generate")


def main(argv=None) -> int:
    # Windows consoles default to cp1252, which can't encode the accented
    # characters in help/log text; force UTF-8 so output never crashes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(
        description="Generate a synthetic Erdos-Renyi g6 corpus.")
    p.add_argument("--vertices", type=int, nargs="+",
                   default=[128, 256, 512, 1024],
                   help="vertex counts, each a power of 2 "
                        "(default: 128 256 512 1024)")
    p.add_argument("-p", "--probability", type=float, nargs="+",
                   default=[round(0.05 * k, 2) for k in range(1, 20)],
                   help="edge probabilities (default: 0.05..0.95 step 0.05)")
    p.add_argument("-n", "--n-graphs", type=int, default=100,
                   help="unique graphs per (n, p) combo (default: 100)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"g6 output dir (default: {DEFAULT_OUT})")
    p.add_argument("--seed", type=int, default=42,
                   help="base random seed (default: 42)")
    p.add_argument("--allow-disconnected", action="store_true",
                   help="keep disconnected graphs (default: reject)")
    p.add_argument("--limit", type=int, default=None,
                   help="write only the first N graphs total (for smoke tests)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    try:
        import networkx  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        log.error("missing dependency: %s. Run `pip install -r requirements.txt`.", e)
        return 2

    bad = [n for n in args.vertices if not is_power_of_two(n)]
    if bad:
        log.error("vertex counts must be powers of 2 (the decomposition maps "
                  "vertices to bitstrings); offending: %s", bad)
        return 2

    out_g6 = args.out / "all.g6"
    manifest_path = args.out / "manifest.json"
    require_connected = not args.allow_disconnected
    params = {
        "vertices": args.vertices,
        "probabilities": args.probability,
        "n_graphs_per_combo": args.n_graphs,
        "base_seed": args.seed,
        "require_connected": require_connected,
        "limit": args.limit,
    }

    stats = Stats()
    generate_corpus(out_g6, args.vertices, args.probability, args.n_graphs,
                    args.seed, stats, require_connected=require_connected,
                    limit=args.limit)
    write_manifest(stats, out_g6, manifest_path, params)

    print("\nDone.")
    print(f"  generated:           {stats.generated}")
    print(f"  skipped_empty:       {stats.skipped_empty}")
    print(f"  skipped_disconnected:{stats.skipped_disconnected}")
    print(f"  skipped_duplicate:   {stats.skipped_duplicate}")
    print(f"  size histogram:     {dict(sorted(stats.size_histogram.items()))}")
    if stats.short_runs:
        print(f"  short runs (got<asked): {stats.short_runs}")
    print(f"  g6 output:          {out_g6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
