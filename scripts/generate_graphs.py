#!/usr/bin/env python3
"""Generate a synthetic g6 corpus (CLI entry point).

One script, two corpora selected with --corpus:

    er         Erdos-Renyi G(n, p) sweep (Pauli-favorable)
    structured counting-path variants (matching-favorable)

Generation logic lives in src/utils/dataset/generate.py.

    python scripts/generate_graphs.py --corpus er                  # ER sweep
    python scripts/generate_graphs.py --corpus structured          # structured
    python scripts/generate_graphs.py --corpus er --limit 100      # smoke test
    python scripts/generate_graphs.py --corpus structured -n 100 --vertices 8 16 32 64 128
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.dataset.generate import (  # noqa: E402
    DEFAULT_ER_OUT,
    DEFAULT_STRUCTURED_OUT,
    Stats,
    generate_er_corpus,
    generate_structured_corpus,
    write_manifest,
    is_power_of_two,
)

log = logging.getLogger("generate")


def main(argv=None) -> int:
    # Windows consoles default to cp1252, which can't encode accented chars.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(
        description="Generate a synthetic g6 corpus (ER or structured).")
    p.add_argument("--corpus", choices=("er", "structured"), required=True,
                   help="which graph family to generate")
    p.add_argument("--vertices", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128],
                   help="vertex counts, each a power of 2 (default: 8 16 32 64 128)")
    p.add_argument("-p", "--probability", type=float, nargs="+",
                   default=[round(0.05 * k, 2) for k in range(1, 20)],
                   help="ER edge probabilities (default: 0.05..0.95 step 0.05; "
                        "ignored for --corpus structured)")
    p.add_argument("-n", "--n-graphs", type=int, default=100,
                   help="unique graphs per cell/size (default: 100)")
    p.add_argument("--out", type=Path, default=None,
                   help="g6 output dir (default: data/<corpus>/g6)")
    p.add_argument("--seed", type=int, default=42,
                   help="base random seed (default: 42)")
    p.add_argument("--allow-disconnected", action="store_true",
                   help="ER only: keep disconnected graphs (default: reject)")
    p.add_argument("--limit", type=int, default=None,
                   help="write only the first N graphs total (smoke tests)")
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
        log.error("vertex counts must be powers of 2; offending: %s", bad)
        return 2

    default_out = DEFAULT_ER_OUT if args.corpus == "er" else DEFAULT_STRUCTURED_OUT
    out_dir = args.out or default_out
    out_g6 = out_dir / "all.g6"
    manifest_path = out_dir / "manifest.json"

    stats = Stats()
    if args.corpus == "er":
        require_connected = not args.allow_disconnected
        params = {
            "corpus": "er",
            "vertices": args.vertices,
            "probabilities": args.probability,
            "n_graphs_per_combo": args.n_graphs,
            "base_seed": args.seed,
            "require_connected": require_connected,
            "limit": args.limit,
        }
        generate_er_corpus(out_g6, args.vertices, args.probability, args.n_graphs,
                           args.seed, stats, require_connected=require_connected,
                           limit=args.limit)
        source = "erdos_renyi_synthetic"
    else:
        params = {
            "corpus": "structured",
            "vertices": args.vertices,
            "n_graphs_per_size": args.n_graphs,
            "base_seed": args.seed,
            "limit": args.limit,
        }
        generate_structured_corpus(out_g6, args.vertices, args.n_graphs,
                                   args.seed, stats, limit=args.limit)
        source = "structured_counting_path"

    write_manifest(stats, out_g6, manifest_path, params, source)

    print("\nDone.")
    print(f"  corpus:              {args.corpus}")
    print(f"  generated:           {stats.generated}")
    print(f"  skipped_duplicate:   {stats.skipped_duplicate}")
    print(f"  skipped_disconnected:{stats.skipped_disconnected}")
    print(f"  size histogram:      {dict(sorted(stats.size_histogram.items()))}")
    if stats.short_runs:
        print(f"  short runs:          {stats.short_runs}")
    print(f"  g6 output:           {out_g6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
