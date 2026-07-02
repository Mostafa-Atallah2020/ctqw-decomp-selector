"""Compute ground-truth decomposition costs (labels) for a g6 corpus.

For each graph in data/g6/all.g6, build the matching and Pauli CTQW circuits
with ctqw-matching-decomp, transpile each to the ["cx", "u3"] basis, and record
the CX count and depth of both. The regression targets follow:

    delta_cx    = cx_pauli    - cx_matching
    delta_depth = depth_pauli - depth_matching

Positive delta means matching is the cheaper decomposition.

This module is the importable library; run it via scripts/label.py.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# corpus in data/g6/, labels in data/labels/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_G6 = _REPO_ROOT / "data" / "er" / "g6" / "all.g6"
DEFAULT_OUT = _REPO_ROOT / "data" / "er" / "labels"

# Fixed labeling parameters (recorded in the manifest; see determinism contract).
N_STEPS = 1
DELTA_T = 0.1
MATCHING_SEED = 99
MATCHING_N_TRIALS = 30
MATCHING_HEURISTIC = "compression_aware"  # XOR-mask grouping; fewer CX than greedy
TRANSPILER_SEED = 0
OPT_LEVEL = 3
BASIS_GATES = ["cx", "u3"]

LABELS_CSV_HEADER = [
    "graph_id",
    "n_vertices",
    "cx_matching",
    "cx_pauli",
    "depth_matching",
    "depth_pauli",
    "delta_cx",       # cx_pauli - cx_matching  (primary target)
    "delta_depth",    # depth_pauli - depth_matching  (secondary target)
    "labeling_time",  # wall-clock seconds to build + transpile both circuits
]

log = logging.getLogger("er.label")


@dataclass
class Stats:
    labeled: int = 0          # newly computed this run
    resumed: int = 0          # skipped because already in labels.csv
    failed: int = 0
    wall_seconds: float = 0.0  # wall-clock for this run's labeling loop
    failures: dict[str, str] = field(default_factory=dict)  # graph_id -> error


def graph6_hash(g6: str) -> str:
    """Stable id for a graph6 string; matches features_g6._graph6_hash."""
    return hashlib.sha1(g6.encode("ascii")).hexdigest()[:16]


def iter_g6_lines(g6_path: Path) -> Iterator[str]:
    with open(g6_path, "r", encoding="ascii") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def _g6_n_vertices(g6: str) -> int:
    """Vertex count from a graph6 string, without building the graph.

    graph6 encodes n in the first byte(s): for n < 63 it is a single byte
    ord(c) - 63. Larger n uses a multi-byte header (prefix '~'); fall back to a
    NetworkX parse there. Our corpus is powers of two up to 1024, so both paths
    occur (n <= 32 single-byte, n >= 64 multi-byte via the ~ header).
    """
    c = ord(g6[0])
    if c != 0x7E:  # '~' marks the multi-byte header
        return c - 63
    import networkx as nx
    return nx.from_graph6_bytes(g6.encode("ascii")).number_of_nodes()


def iter_g6_by_vertices(g6_path: Path) -> Iterator[str]:
    """Yield g6 lines ordered by ascending vertex count (ties keep file order).

    Parsing every line up front is cheap; it is labeling that is expensive. This
    lets the labeler do the many cheap small graphs first and the few costly
    large ones last, so an interruption loses the least work.
    """
    lines = list(iter_g6_lines(g6_path))
    lines.sort(key=_g6_n_vertices)  # stable sort preserves file order within an n
    yield from lines


def _transpiled_metrics(circuit, transpile, seed_transpiler):
    """Return (cx_count, depth) for a circuit transpiled to the fixed basis."""
    t = transpile(circuit, basis_gates=BASIS_GATES,
                  optimization_level=OPT_LEVEL, seed_transpiler=seed_transpiler)
    return t.count_ops().get("cx", 0), t.depth()


def label_one(g6: str) -> dict:
    """Compute the label row for a single graph6 string.

    Raises on any decomposition/transpile error so the caller can record it.
    """
    import time
    from qiskit import transpile
    from ctqw_matching_decomp.core import (
        MultiEdgeGraph, MatchingDecomposition, PauliDecomposition,
    )
    from ctqw_matching_decomp.utils.graph.g6_utils import g6_to_edge_set

    edges = g6_to_edge_set(g6)
    if not edges:
        raise ValueError("empty or unparseable g6")
    G = MultiEdgeGraph(edges)

    t0 = time.perf_counter()
    m_circ = MatchingDecomposition(
        G, heuristic=MATCHING_HEURISTIC, n_trials=MATCHING_N_TRIALS,
        seed=MATCHING_SEED,
    ).build_circuit(n_steps=N_STEPS, delta_t=DELTA_T)
    cx_m, depth_m = _transpiled_metrics(m_circ, transpile, TRANSPILER_SEED)

    p_circ = PauliDecomposition(G).build_circuit(n_steps=N_STEPS, delta_t=DELTA_T)
    cx_p, depth_p = _transpiled_metrics(p_circ, transpile, TRANSPILER_SEED)
    labeling_time = time.perf_counter() - t0

    return {
        "graph_id": graph6_hash(g6),
        "n_vertices": G.n_vertices,
        "cx_matching": cx_m,
        "cx_pauli": cx_p,
        "depth_matching": depth_m,
        "depth_pauli": depth_p,
        "delta_cx": cx_p - cx_m,
        "delta_depth": depth_p - depth_m,
        "labeling_time": round(labeling_time, 3),
    }


def load_done_ids(labels_csv: Path) -> set[str]:
    """graph_ids already present in labels.csv (for resume)."""
    if not labels_csv.exists():
        return set()
    done: set[str] = set()
    with open(labels_csv, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            gid = row.get("graph_id")
            if gid:
                done.add(gid)
    return done


def _fmt_duration(seconds: float) -> str:
    """Duration as hours/minutes/seconds, e.g. '2h 05m 09s'."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def label_corpus(g6_path: Path, labels_csv: Path, stats: Stats,
                 limit: int | None = None) -> None:
    """Label every graph in g6_path, appending to labels_csv. Resumable.

    Graphs are processed in ascending vertex count (cheap first). The live
    progress line shows elapsed time and an ETA computed per vertex count:
    remaining graphs at each n times the measured average time at that n, which
    stays accurate even though the per-graph cost rises sharply with n.
    """
    import time

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(labels_csv)
    stats.resumed = len(done)

    # Materialize the vertex-sorted work list once, so we know what remains at
    # each n for the ETA. Skip graphs already labelled.
    ordered = [g6 for g6 in iter_g6_by_vertices(g6_path)
               if graph6_hash(g6) not in done]
    if limit is not None:
        ordered = ordered[:limit]

    remaining_by_n: dict[int, int] = {}
    for g6 in ordered:
        n = _g6_n_vertices(g6)
        remaining_by_n[n] = remaining_by_n.get(n, 0) + 1

    total_new = len(ordered)
    log.info("labeling %s -> %s (%d done, %d to do; sizes %s)",
             g6_path, labels_csv, len(done), total_new,
             dict(sorted(remaining_by_n.items())))

    # Running average time per n, for the ETA.
    n_time_sum: dict[int, float] = {}
    n_time_cnt: dict[int, int] = {}

    def eta_seconds() -> float:
        total = 0.0
        for n, left in remaining_by_n.items():
            if left <= 0:
                continue
            if n_time_cnt.get(n):
                avg = n_time_sum[n] / n_time_cnt[n]
            else:
                # No measurement yet at this n: extrapolate from the largest n
                # we have timed (cost grows with n, so this is a floor).
                measured = [(k, n_time_sum[k] / n_time_cnt[k])
                            for k in n_time_cnt]
                avg = max((a for _, a in measured), default=1.0)
            total += left * avg
        return total

    new_file = not labels_csv.exists()
    t_start = time.perf_counter()
    with open(labels_csv, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABELS_CSV_HEADER)
        if new_file:
            writer.writeheader()
            fh.flush()

        for g6 in ordered:
            gid = graph6_hash(g6)
            try:
                row = label_one(g6)
            except Exception as e:  # record and keep going
                stats.failed += 1
                stats.failures[gid] = f"{type(e).__name__}: {e}"
                log.warning("label failed for %s: %s", gid, e)
                continue
            writer.writerow(row)
            fh.flush()  # per-row flush so a kill loses at most one graph
            done.add(gid)
            stats.labeled += 1

            n = row["n_vertices"]
            remaining_by_n[n] = remaining_by_n.get(n, 0) - 1
            n_time_sum[n] = n_time_sum.get(n, 0.0) + row["labeling_time"]
            n_time_cnt[n] = n_time_cnt.get(n, 0) + 1

            elapsed = time.perf_counter() - t_start
            # total_at_n = done-so-far + still-left, so we can show "k/K at n".
            done_at_n = n_time_cnt[n]
            total_at_n = done_at_n + remaining_by_n[n]
            print(
                f"\rtotal {stats.labeled:>5}/{total_new:<5}"
                f" | n={n:<4} {done_at_n:>4}/{total_at_n:<4}"
                f" | {row['labeling_time']:5.1f}s/graph"
                f" | elapsed {_fmt_duration(elapsed)}"
                f" | ETA {_fmt_duration(eta_seconds())}   ",
                end="", flush=True,
            )
    print()  # finish the live line

    stats.wall_seconds = round(time.perf_counter() - t_start, 1)
    log.info("done. labeled=%d resumed=%d failed=%d wall=%s",
             stats.labeled, stats.resumed, stats.failed,
             _fmt_duration(stats.wall_seconds))


def label_params() -> dict:
    """The fixed labeling parameters, for the manifest."""
    return {
        "n_steps": N_STEPS,
        "delta_t": DELTA_T,
        "matching_heuristic": MATCHING_HEURISTIC,
        "matching_n_trials": MATCHING_N_TRIALS,
        "matching_seed": MATCHING_SEED,
        "transpiler_seed": TRANSPILER_SEED,
        "optimization_level": OPT_LEVEL,
        "basis_gates": BASIS_GATES,
    }


def write_manifest(stats: Stats, g6_path: Path, labels_csv: Path,
                   manifest_path: Path) -> None:
    import json
    payload = {
        "source": "erdos_renyi_synthetic",
        "g6_input": str(g6_path),
        "labels_csv": str(labels_csv),
        "labels_csv_header": LABELS_CSV_HEADER,
        "params": label_params(),
        "stats": {
            "labeled": stats.labeled,
            "resumed": stats.resumed,
            "failed": stats.failed,
            "wall_seconds": stats.wall_seconds,
            "failures": stats.failures,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("wrote manifest to %s", manifest_path)
