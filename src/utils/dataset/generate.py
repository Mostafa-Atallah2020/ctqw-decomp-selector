"""Generate a synthetic Erdős-Rényi g6 corpus for CTQW cost labeling.

Sweeps Erdős-Rényi G(n, p) graphs across powers-of-two vertex counts and a
range of edge probabilities, writing one g6 line per graph to data/g6/all.g6.

The generation routine is adapted from
https://github.com/Mostafa-Atallah2020/ctqw-matching-decomp
(analysis/generate_erdos_renyi_graphs.py).

Vertex counts must be powers of two: each vertex is mapped to a bitstring over
n_qubits = log2(n) for the CTQW Hamiltonian, which spans a full 2**n_qubits
vertex space.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

# g6 lands in data/g6/, the same location the feature extractor reads from.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "data" / "g6"

log = logging.getLogger("er.generate")


@dataclass
class Stats:
    generated: int = 0
    skipped_empty: int = 0
    skipped_disconnected: int = 0
    skipped_duplicate: int = 0
    short_runs: dict[str, int] = field(default_factory=dict)  # "n@p" -> got<requested
    size_histogram: dict[int, int] = field(default_factory=dict)
    cell_counts: dict[str, int] = field(default_factory=dict)  # "n@p" -> realized

    def record(self, n: int, p: float | None = None) -> None:
        self.generated += 1
        self.size_histogram[n] = self.size_histogram.get(n, 0) + 1
        if p is not None:
            key = f"{n}@{p}"
            self.cell_counts[key] = self.cell_counts.get(key, 0) + 1

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def generate_erdos_renyi_graphs(n_vertices, n_graphs, p, seed=None,
                                require_connected=True,
                                stats: "Stats | None" = None):
    """Generate unique Erdős-Rényi random graphs G(n, p).

    Each edge is included independently with probability p; disconnected graphs
    are rejected when require_connected is True; isomorphic duplicates are
    rejected via the Weisfeiler-Lehman graph hash; empty graphs are skipped.
    Returns a list of NetworkX graphs (<= n_graphs).
    """
    import networkx as nx
    import numpy as np

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    graphs = []
    seen_hashes: set[str] = set()
    attempts = 0
    max_attempts = n_graphs * 100

    while len(graphs) < n_graphs and attempts < max_attempts:
        attempts += 1

        G = nx.erdos_renyi_graph(n_vertices, p, seed=random.randint(0, 2**31))

        # Skip empty graphs.
        if G.number_of_edges() == 0:
            if stats is not None:
                stats.skipped_empty += 1
            continue

        # Reject disconnected graphs. At the low-p end on large n this is rare
        # since we sit above the connectivity threshold ~ln(n)/n; at high p
        # essentially every graph is connected.
        if require_connected and not nx.is_connected(G):
            if stats is not None:
                stats.skipped_disconnected += 1
            continue

        # Reject isomorphic duplicates via graph hash.
        try:
            h = nx.weisfeiler_lehman_graph_hash(G)
            if h in seen_hashes:
                if stats is not None:
                    stats.skipped_duplicate += 1
                continue
            seen_hashes.add(h)
        except Exception:
            pass

        graphs.append(G)

    if len(graphs) < n_graphs:
        log.warning("n=%d p=%s: only %d/%d unique graphs after %d attempts",
                    n_vertices, p, len(graphs), n_graphs, attempts)
        if stats is not None:
            stats.short_runs[f"{n_vertices}@{p}"] = len(graphs)

    return graphs[:n_graphs]


def generate_corpus(out_g6: Path, vertices: list[int], probabilities: list[float],
                    n_graphs: int, base_seed: int, stats: Stats,
                    require_connected: bool = True,
                    limit: int | None = None) -> None:
    """Sweep (n, p), write all graphs as one g6 line each to out_g6."""
    import networkx as nx

    out_g6.parent.mkdir(parents=True, exist_ok=True)
    log.info("generating ER corpus -> %s", out_g6)

    combos = [(n, p) for n in vertices for p in probabilities]
    n_cells = len(combos)

    written = 0
    with open(out_g6, "w", encoding="ascii", newline="\n") as out_fh:
        for cell_idx, (n_vertices, p) in enumerate(combos, start=1):
            # Per-(n,p) seed, so the
            # same combo reproduces the same graphs regardless of sweep order.
            seed = base_seed + int(p * 1000) + n_vertices if base_seed else None
            graphs = generate_erdos_renyi_graphs(
                n_vertices, n_graphs, p, seed=seed,
                require_connected=require_connected, stats=stats)
            cell_written = 0
            for g in graphs:
                if limit is not None and written >= limit:
                    # Flush before returning so the partial file is on disk.
                    out_fh.flush()
                    print()  # end the live line
                    log.info("hit --limit %d, stopping", limit)
                    return
                g6 = nx.to_graph6_bytes(g, header=False).decode("ascii").strip()
                out_fh.write(g6 + "\n")
                stats.record(n_vertices, p)
                written += 1
                cell_written += 1
                # Live single-line update, rewritten on every graph written.
                print(f"\rcell {cell_idx}/{n_cells}  n={n_vertices} p={p}  "
                      f"{cell_written}/{n_graphs} in cell  "
                      f"{written} total", end="", flush=True)
            # Flush to disk at each cell boundary so a kill leaves a usable file.
            out_fh.flush()
    print()  # finish the live line before the summary

    log.info("done. generated=%d skipped_empty=%d skipped_disconnected=%d "
             "skipped_duplicate=%d", stats.generated, stats.skipped_empty,
             stats.skipped_disconnected, stats.skipped_duplicate)


def write_manifest(stats: Stats, out_g6: Path, manifest_path: Path,
                   params: dict) -> None:
    payload = {
        "source": "erdos_renyi_synthetic",
        "params": params,
        "g6_output": str(out_g6),
        "g6_output_size_bytes": (out_g6.stat().st_size
                                 if out_g6.exists() else None),
        "stats": stats.as_dict(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("wrote manifest to %s", manifest_path)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
