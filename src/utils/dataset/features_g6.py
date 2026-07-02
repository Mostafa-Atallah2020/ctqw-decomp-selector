"""Extract per-graph topology features from a g6 corpus to a single CSV.

Reads data/g6/all.g6 (one graph6 string per line, produced by the generator),
reconstructs each NetworkX graph, and writes one CSV row per graph with the
topology features the model trains on. Rows are keyed by graph6 hash, so the
key is the graph itself and is stable across runs.

Feature values come from utils.graph.properties.calculate_graph_properties.
There is only a topology CSV; the synthetic graphs carry no composition,
descriptors, or other chemistry.

Three features are deliberately NOT extracted here because their enumerations
are exponential in the worst case and hang on large dense graphs:
automorphism_group_size and orbit_count (NetworkX VF2 self-isomorphism), and
clique_number (maximal-clique enumeration). On dense n=128 ER graphs these can
run for minutes or never finish, and a thread timeout cannot bound the
automorphism one (the stuck worker holds the GIL). Every remaining feature is
sub-second even at n=128 (Laplacian eigendecomposition, the slowest, is ~0.8 s).
The automorphism features are also near degenerate for ER (graphs are almost
always rigid, |Aut|=1). If symmetry or clique features are needed later, compute
automorphism with pynauty (nauty), which handles n=128 in milliseconds. They are
disabled via the with_automorphism / with_clique flags on
calculate_graph_properties.

This module is the importable library; run it via scripts/extract_features.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from utils.graph.properties import calculate_graph_properties

# g6 corpus lives in data/g6/, derived features in data/features/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_G6 = _REPO_ROOT / "data" / "g6" / "all.g6"
DEFAULT_OUT = _REPO_ROOT / "data" / "features"

# Graph-topology feature columns, keyed by graph6 hash.
GRAPH_CSV_HEADER = [
    "graph_id",       # graph6 hash (stable per topology)
    "n_vertices",
    # size / sparsity
    "edge_count",
    "edge_density",
    # bipartite / cycles
    "is_bipartite",
    "is_tree",
    "cycle_count",
    "triangle_count",
    # diameter
    "diameter",
    # degree stats
    "min_degree",
    "max_degree",
    "avg_degree",
    "degree_variance",
    "is_regular",
    # clustering / spectral
    "avg_clustering",
    "spectral_gap",
    # matching-decomp proxies (paper-motivated)
    "max_matching_size",
    "chromatic_index_lower_bound",
]

log = logging.getLogger("er.features")


@dataclass
class Stats:
    parsed: int = 0
    skipped_parse_error: int = 0
    skipped_disconnected: int = 0
    duplicate_graph: int = 0


def _graph6_hash(g6: str) -> str:
    """Stable short id for a graph6 string (its topology)."""
    return hashlib.sha1(g6.encode("ascii")).hexdigest()[:16]


def iter_g6_lines(g6_path: Path) -> Iterator[str]:
    """Yield each non-empty, stripped graph6 line from the corpus file."""
    with open(g6_path, "r", encoding="ascii") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def _bool_to_int(v: object) -> int:
    return 1 if v else 0


def _graph_row(graph_id: str, graph) -> dict:
    """Build the graph-features row for one graph.

    Skips the automorphism features (see module note); all other features come
    from calculate_graph_properties and are cheap even at n=128.
    """
    p = calculate_graph_properties(graph, with_automorphism=False,
                                   with_clique=False)
    return {
        "graph_id": graph_id,
        "n_vertices": graph.number_of_nodes(),
        "edge_count": p["edge_count"],
        "edge_density": round(p["edge_density"], 6),
        "is_bipartite": _bool_to_int(p["is_bipartite"]),
        "is_tree": _bool_to_int(p["is_tree"]),
        "cycle_count": p["cycle_count"] if p["cycle_count"] is not None else "",
        "triangle_count": p["triangle_count"],
        "diameter": p["diameter"] if p["diameter"] is not None else "",
        "min_degree": p["min_degree"],
        "max_degree": p["max_degree"],
        "avg_degree": round(p["avg_degree"], 6),
        "degree_variance": round(p["degree_variance"], 6),
        "is_regular": _bool_to_int(p["is_regular"]),
        "avg_clustering": round(p["avg_clustering"], 6),
        "spectral_gap": round(p["spectral_gap"], 6),
        "max_matching_size": p["max_matching_size"],
        "chromatic_index_lower_bound": p["chromatic_index_lower_bound"],
    }


def extract_features(g6_path: Path, graph_csv: Path, stats: Stats,
                     limit: int | None = None) -> None:
    import networkx as nx

    graph_csv.parent.mkdir(parents=True, exist_ok=True)
    log.info("reading g6 corpus  <- %s", g6_path)
    log.info("writing graph features -> %s", graph_csv)

    seen_ids: set[str] = set()
    total = 0
    with open(graph_csv, "w", encoding="utf-8", newline="") as gh:
        gw = csv.DictWriter(gh, fieldnames=GRAPH_CSV_HEADER)
        gw.writeheader()

        for i, g6 in enumerate(iter_g6_lines(g6_path)):
            if limit is not None and i >= limit:
                break
            try:
                g = nx.from_graph6_bytes(g6.encode("ascii"))
            except Exception as e:
                log.debug("parse error on line %d: %s", i + 1, e)
                stats.skipped_parse_error += 1
                continue
            if not nx.is_connected(g):
                # generate.py rejects these, but guard in case of a stray line
                stats.skipped_disconnected += 1
                continue

            graph_id = _graph6_hash(g6)
            if graph_id in seen_ids:
                stats.duplicate_graph += 1
                continue
            seen_ids.add(graph_id)

            gw.writerow(_graph_row(graph_id, g))
            stats.parsed += 1
            total += 1
            print(f"\r{total} graphs  (n={g.number_of_nodes()})",
                  end="", flush=True)
    print()  # finish the live line

    log.info("done. parsed=%d skipped_parse_error=%d skipped_disconnected=%d "
             "duplicate_graph=%d", stats.parsed, stats.skipped_parse_error,
             stats.skipped_disconnected, stats.duplicate_graph)


def write_manifest(stats: Stats, g6_path: Path, graph_csv: Path,
                   manifest_path: Path) -> None:
    payload = {
        "source": "erdos_renyi_synthetic",
        "g6_input": str(g6_path),
        "graph_csv": str(graph_csv),
        "graph_csv_size_bytes": (graph_csv.stat().st_size
                                 if graph_csv.exists() else None),
        "graph_csv_header": GRAPH_CSV_HEADER,
        "stats": stats.__dict__,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("wrote manifest to %s", manifest_path)


