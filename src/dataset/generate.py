"""Generate synthetic g6 datasets for CTQW cost labeling.

Two graph families, kept as separate datasets:

- Erdos-Renyi G(n, p) (generate_er_dataset): random graphs over a density grid.
  Their random edge Hamming distributions leave matching little to compress, so
  Pauli usually wins.
- Structured counting-path (generate_structured_dataset): counting paths and
  variants (XOR-permuted, extra-edges, segment-reversed, perturbed) whose mixed
  Hamming structure matching exploits, so matching usually wins.

Both write one g6 line per graph plus a manifest, dedup by Weisfeiler-Lehman
hash, keep only connected graphs, and require power-of-two vertex counts (each
vertex is a bitstring over log2(n) qubits).

Importable library. Run it via `scripts/generate_graphs.py --dataset er|structured`.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ER_OUT = _REPO_ROOT / "data" / "er" / "g6"
DEFAULT_STRUCTURED_OUT = _REPO_ROOT / "data" / "structured" / "g6"

log = logging.getLogger("generate")


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    generated: int = 0
    skipped_empty: int = 0
    skipped_disconnected: int = 0
    skipped_duplicate: int = 0
    short_runs: dict[str, int] = field(default_factory=dict)
    size_histogram: dict[int, int] = field(default_factory=dict)
    # cell key is "n@p" for ER, "n@variant" for structured.
    cell_counts: dict[str, int] = field(default_factory=dict)

    def record(self, n: int, cell: "str | float | None" = None) -> None:
        self.generated += 1
        self.size_histogram[n] = self.size_histogram.get(n, 0) + 1
        if cell is not None:
            key = f"{n}@{cell}"
            self.cell_counts[key] = self.cell_counts.get(key, 0) + 1

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def load_existing_g6(out_g6: Path) -> "tuple[set[str], int]":
    """Read an existing g6 dataset and return (WL-hash set, line count).

    Makes generation RESUMABLE and IDEMPOTENT: a re-run seeds the dedup set with
    graphs already on disk and appends only new ones. Absent file -> (set(), 0).
    """
    import networkx as nx

    if not out_g6.exists():
        return set(), 0
    hashes: set[str] = set()
    count = 0
    with open(out_g6, "r", encoding="ascii") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                G = nx.from_graph6_bytes(line.encode("ascii"))
                hashes.add(nx.weisfeiler_lehman_graph_hash(G))
            except Exception:
                # Unparseable line: keep it in the file but don't hash (won't dedup).
                pass
    log.info("resuming from existing dataset: %d graphs already in %s",
             count, out_g6)
    return hashes, count


def write_manifest(stats: Stats, out_g6: Path, manifest_path: Path,
                   params: dict, source: str) -> None:
    payload = {
        "source": source,
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


# ---------------------------------------------------------------------------
# Erdos-Renyi
# ---------------------------------------------------------------------------

def generate_erdos_renyi_graphs(n_vertices, n_graphs, p, seed=None,
                                require_connected=True,
                                stats: "Stats | None" = None):
    """Generate unique Erdős-Rényi random graphs G(n, p).

    Each edge included independently with probability p. Rejects disconnected
    (when require_connected), WL-hash duplicates, and empty graphs. Returns a
    list of NetworkX graphs (<= n_graphs).
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

        if G.number_of_edges() == 0:
            if stats is not None:
                stats.skipped_empty += 1
            continue

        # Reject disconnected graphs (rare: we sit above the ~ln(n)/n threshold).
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


def generate_er_dataset(out_g6: Path, vertices: list[int],
                       probabilities: list[float], n_graphs: int,
                       base_seed: int, stats: Stats,
                       require_connected: bool = True,
                       limit: int | None = None) -> None:
    """Sweep ER (n, p), appending new graphs (one g6 line each) to out_g6.

    RESUMABLE: existing graphs are loaded and regenerated duplicates skipped, so
    a re-run only adds graphs instead of overwriting the dataset.
    """
    import networkx as nx

    out_g6.parent.mkdir(parents=True, exist_ok=True)
    log.info("generating ER dataset -> %s", out_g6)

    preexisting, written = load_existing_g6(out_g6)

    combos = [(n, p) for n in vertices for p in probabilities]
    n_cells = len(combos)

    with open(out_g6, "a", encoding="ascii", newline="\n") as out_fh:
        for cell_idx, (n_vertices, p) in enumerate(combos, start=1):
            # Per-(n,p) seed: same combo reproduces the same graphs regardless of order.
            seed = base_seed + int(p * 1000) + n_vertices if base_seed else None
            graphs = generate_erdos_renyi_graphs(
                n_vertices, n_graphs, p, seed=seed,
                require_connected=require_connected, stats=stats)
            cell_written = 0
            for g in graphs:
                if limit is not None and written >= limit:
                    out_fh.flush()
                    print()
                    log.info("hit --limit %d, stopping", limit)
                    return
                # Skip graphs already on disk (idempotent re-runs).
                try:
                    if nx.weisfeiler_lehman_graph_hash(g) in preexisting:
                        stats.skipped_duplicate += 1
                        continue
                except Exception:
                    pass
                g6 = nx.to_graph6_bytes(g, header=False).decode("ascii").strip()
                out_fh.write(g6 + "\n")
                stats.record(n_vertices, p)
                written += 1
                cell_written += 1
                print(f"\rcell {cell_idx}/{n_cells}  n={n_vertices} p={p}  "
                      f"{cell_written}/{n_graphs} in cell  "
                      f"{written} total", end="", flush=True)
            out_fh.flush()  # per-cell flush so a kill leaves a usable file
    print()

    log.info("done. generated=%d skipped_empty=%d skipped_disconnected=%d "
             "skipped_duplicate=%d", stats.generated, stats.skipped_empty,
             stats.skipped_disconnected, stats.skipped_duplicate)


# Structured counting-path edge-set builders: each returns a set of (u, v)
# bitstring-tuple edges, u < v lexicographically.

def _n_bits(n_vertices: int) -> int:
    return max(1, (n_vertices - 1).bit_length())


def _edges_from_order(order, n_bits) -> set:
    edges = set()
    for i in range(len(order) - 1):
        u = format(order[i], f"0{n_bits}b")
        v = format(order[i + 1], f"0{n_bits}b")
        edges.add((min(u, v), max(u, v)))
    return edges


def counting_path(n_vertices, offset=0, reverse=False) -> set:
    """Base counting path 0-1-2-...-(n-1), optionally rotated and/or reversed."""
    b = _n_bits(n_vertices)
    order = list(range(n_vertices))
    order = order[offset:] + order[:offset]
    if reverse:
        order = order[::-1]
    return _edges_from_order(order, b)


def xor_permuted_path(n_vertices, xor_mask) -> set:
    """Path over vertices XOR'd with a mask (relabels while keeping structure)."""
    b = _n_bits(n_vertices)
    permuted = [i ^ xor_mask for i in range(n_vertices)]
    return _edges_from_order(permuted, b)


def perturbed_path(n_vertices, n_swaps=1, seed=None) -> set:
    """Counting path with n_swaps random adjacent transpositions."""
    if seed is not None:
        random.seed(seed)
    order = list(range(n_vertices))
    for _ in range(n_swaps):
        if len(order) >= 3:
            i = random.randint(1, len(order) - 2)
            order[i], order[i + 1] = order[i + 1], order[i]
    return _edges_from_order(order, _n_bits(n_vertices))


def segment_reversed_path(n_vertices, segment_size=4, seed=None) -> set:
    """Counting path with some length-segment_size segments reversed."""
    if seed is not None:
        random.seed(seed)
    order = list(range(n_vertices))
    for start in range(0, n_vertices, segment_size):
        end = min(start + segment_size, n_vertices)
        if random.random() > 0.5:
            order[start:end] = list(reversed(order[start:end]))
    return _edges_from_order(order, _n_bits(n_vertices))


def path_with_extra_edges(n_vertices, base_edges, n_extra=1, seed=None) -> set:
    """Add n_extra random non-path edges to a base path (introduces cycles)."""
    if seed is not None:
        random.seed(seed)
    b = _n_bits(n_vertices)
    edges = set(base_edges)
    candidates = []
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            e = (format(i, f"0{b}b"), format(j, f"0{b}b"))
            e = (min(e), max(e))
            if e not in edges:
                candidates.append(e)
    random.shuffle(candidates)
    for e in candidates[:n_extra]:
        edges.add(e)
    return edges


def _edges_to_graph(edges: set):
    """Bitstring edge set -> integer-labelled NetworkX graph on 2**n_bits nodes."""
    import networkx as nx
    nb = len(next(iter(edges))[0])
    n_vertices = 2 ** nb
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    for u, v in edges:
        G.add_edge(int(u, 2), int(v, 2))
    return G


def generate_structured_graphs(n_vertices, n_graphs, seed=None,
                               stats: "Stats | None" = None):
    """Generate up to n_graphs unique connected structured graphs for one size.

    Cycles through variant families (counting-path offsets/reversals, XOR-perms,
    extra-edge paths, then segment-reversed and perturbed for boundary
    diversity), deduped by WL hash. Returns (graph, variant_name) pairs.
    """
    import networkx as nx

    if seed is not None:
        random.seed(seed)

    graphs = []
    seen: set[str] = set()

    def add(edges, variant):
        if not edges:
            return
        G = _edges_to_graph(edges)
        if not nx.is_connected(G):
            if stats is not None:
                stats.skipped_disconnected += 1
            return
        h = nx.weisfeiler_lehman_graph_hash(G)
        if h in seen:
            if stats is not None:
                stats.skipped_duplicate += 1
            return
        seen.add(h)
        graphs.append((G, variant))

    base = counting_path(n_vertices)

    # 1) counting-path rotations and reversals
    for offset in range(n_vertices):
        for reverse in (False, True):
            if len(graphs) >= n_graphs:
                break
            add(counting_path(n_vertices, offset=offset, reverse=reverse),
                "counting_path")

    # 2) XOR-permuted paths (one per mask)
    for mask in range(1, n_vertices):
        if len(graphs) >= n_graphs:
            break
        add(xor_permuted_path(n_vertices, mask), "xor_permuted")

    # 3) extra-edge paths - strongly matching-favorable, so fill from these first.
    k = 0
    while len(graphs) < n_graphs and k < n_graphs * 6:
        add(path_with_extra_edges(n_vertices, base,
                                  n_extra=1 + (k % 4), seed=(seed or 0) + 1000 + k),
            "extra_edges")
        k += 1

    # 4) segment-reversed then lightly-perturbed paths for boundary diversity.
    k = 0
    while len(graphs) < n_graphs and k < n_graphs * 4:
        s = (seed or 0) + k
        add(segment_reversed_path(n_vertices, segment_size=4, seed=s),
            "segment_reversed")
        add(perturbed_path(n_vertices, n_swaps=1, seed=s + 1), "perturbed")
        k += 1

    if len(graphs) < n_graphs:
        log.warning("n=%d: only %d/%d unique structured graphs",
                    n_vertices, len(graphs), n_graphs)
        if stats is not None:
            stats.short_runs[str(n_vertices)] = len(graphs)

    return graphs[:n_graphs]


def generate_structured_dataset(out_g6: Path, vertices, n_graphs, base_seed,
                               stats: Stats, limit: int | None = None) -> None:
    """Generate structured graphs for each vertex count, appending to out_g6.

    RESUMABLE: existing graphs are loaded and regenerated duplicates skipped, so
    a re-run only adds graphs instead of overwriting the dataset.
    """
    import networkx as nx

    out_g6.parent.mkdir(parents=True, exist_ok=True)
    log.info("generating structured dataset -> %s", out_g6)

    preexisting, written = load_existing_g6(out_g6)

    with open(out_g6, "a", encoding="ascii", newline="\n") as fh:
        for n_vertices in vertices:
            seed = base_seed + n_vertices if base_seed else None
            graphs = generate_structured_graphs(n_vertices, n_graphs, seed=seed,
                                                 stats=stats)
            for i, (G, variant) in enumerate(graphs):
                if limit is not None and written >= limit:
                    fh.flush()
                    print()
                    log.info("hit --limit %d, stopping", limit)
                    return
                # Skip graphs already on disk (idempotent re-runs).
                try:
                    if nx.weisfeiler_lehman_graph_hash(G) in preexisting:
                        stats.skipped_duplicate += 1
                        continue
                except Exception:
                    pass
                g6 = nx.to_graph6_bytes(G, header=False).decode("ascii").strip()
                fh.write(g6 + "\n")
                stats.record(n_vertices, variant)
                written += 1
                print(f"\rn={n_vertices}  {i + 1}/{len(graphs)} in size  "
                      f"{written} total", end="", flush=True)
            fh.flush()
    print()

    log.info("done. generated=%d skipped_duplicate=%d skipped_disconnected=%d",
             stats.generated, stats.skipped_duplicate, stats.skipped_disconnected)
