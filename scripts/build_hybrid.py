#!/usr/bin/env python3
"""Build a balanced hybrid training set from the er and structured corpora (CLI).

Selects graph_ids so that within each vertex count the matching-wins and
Pauli-wins are equal (leakage-safe: no size/corpus shortcut). Writes
data/hybrid/{features,labels}/ with the same schema as the source corpora.
Logic lives in src/utils/dataset/hybrid.py.

    python scripts/build_hybrid.py
    python scripts/build_hybrid.py --seed 1 --out F:/data/hybrid
"""

import argparse
import time
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.progress import fmt_duration
from utils.dataset.hybrid import DEFAULT_DATA, DEFAULT_OUT, build  # noqa: E402

log = logging.getLogger("hybrid")


def main(argv=None) -> int:
    _t0 = time.perf_counter()
    p = argparse.ArgumentParser(
        description="Build a within-n class-balanced hybrid training set.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help=f"data root (default: {DEFAULT_DATA})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"hybrid output dir (default: {DEFAULT_OUT})")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    for corpus in ("er", "structured"):
        if not (args.data / corpus / "labels" / "labels.csv").exists():
            log.error("missing labels for '%s'. Label it first.", corpus)
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
              f"({per_size}; {unl['per_corpus']})")

    print(f"  -> {args.out}")
    print(f"  done in {fmt_duration(time.perf_counter() - _t0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
