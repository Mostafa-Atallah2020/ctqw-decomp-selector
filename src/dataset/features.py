"""Extract per-graph topology features from a g6 dataset to a single CSV.

Reads a g6 file (one graph6 string per line), reconstructs each NetworkX graph,
and writes one CSV row per graph, keyed by a stable graph6 hash. Values come
from graph.properties.calculate_graph_properties.

Automorphism_group_size, orbit_count, and clique_number are deliberately NOT
extracted: their enumerations are worst-case exponential and hang on large dense
graphs (the automorphism worker holds the GIL, so a timeout can't bound it), and
are near-degenerate for ER (|Aut|=1). Disabled via with_automorphism /
with_clique on calculate_graph_properties.

Importable library. Run it via scripts/extract_features.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import networkx as nx
from scipy.linalg import eigvalsh

from graph.properties import calculate_graph_properties

# g6 dataset lives in data/g6/, derived features in data/features/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_G6 = _REPO_ROOT / "data" / "er" / "g6" / "all.g6"
DEFAULT_OUT = _REPO_ROOT / "data" / "er" / "features"

# Graph-topology feature columns, keyed by graph6 hash. Collinear features are
# excluded (edge_count, cycle_count, chromatic_index_lower_bound).
GRAPH_CSV_HEADER = [
    "graph_id",       # graph6 hash (stable per topology)
    "n_vertices",
    # sparsity
    "edge_density",
    # bipartite / cycles
    "is_bipartite",
    "is_tree",
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
    # matching-decomp proxy (paper-motivated)
    "max_matching_size",
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
    """Yield each non-empty, stripped graph6 line from the dataset file."""
    with open(g6_path, "r", encoding="ascii") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def _bool_to_int(v: object) -> int:
    return 1 if v else 0


def _graph_row(graph_id: str, graph) -> dict:
    """Build the graph-features row for one graph (automorphism features skipped,
    see module note)."""
    p = calculate_graph_properties(graph, with_automorphism=False,
                                   with_clique=False)
    return {
        "graph_id": graph_id,
        "n_vertices": graph.number_of_nodes(),
        "edge_density": round(p["edge_density"], 6),
        "is_bipartite": _bool_to_int(p["is_bipartite"]),
        "is_tree": _bool_to_int(p["is_tree"]),
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
    }


def _existing_feature_ids(graph_csv: Path) -> "set[str]":
    """graph_ids already present in an existing feature CSV (empty if absent).

    Lets extraction RESUME: already-computed rows are kept, so a re-run only
    featurizes graphs new to the g6 dataset.
    """
    if not graph_csv.exists():
        return set()
    ids: set[str] = set()
    with open(graph_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gid = row.get("graph_id")
            if gid:
                ids.add(gid)
    return ids


def extract_features(g6_path: Path, graph_csv: Path, stats: Stats,
                     limit: int | None = None) -> None:
    import networkx as nx

    graph_csv.parent.mkdir(parents=True, exist_ok=True)
    log.info("reading g6 dataset  <- %s", g6_path)

    # Resume: keep rows already computed, append only graphs new to the dataset.
    seen_ids: set[str] = _existing_feature_ids(graph_csv)
    is_new_file = len(seen_ids) == 0
    if not is_new_file:
        log.info("resuming: %d graphs already featurized in %s",
                 len(seen_ids), graph_csv)
    log.info("%s graph features -> %s",
             "writing" if is_new_file else "appending", graph_csv)

    total = 0
    mode = "w" if is_new_file else "a"
    with open(graph_csv, mode, encoding="utf-8", newline="") as gh:
        gw = csv.DictWriter(gh, fieldnames=GRAPH_CSV_HEADER)
        if is_new_file:
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
            print(f"\r{total} new graphs  (n={g.number_of_nodes()})",
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


# --- features constants ------------------------------------------------------
HEAT_TIMES = [0.5, 1.0, 2.0, 5.0]      # diffusion times for the heat-kernel traces

#: Coefficients below this magnitude are treated as absent from the decomposition.
_PAULI_ATOL = 1e-12

#: Matching-decomposition settings. MUST match the labeler's (src/dataset/label.py)
#: so feature counts describe the same decomposition whose CX cost made the label.
_MATCH_HEURISTIC = "compression_aware"
_MATCH_N_TRIALS = 30
_MATCH_SEED = 99


def spectral_rich(G: "nx.Graph") -> dict:
    """Normalized-Laplacian spectrum plus heat-kernel diffusion signatures. Pure
    graph features (only how a walk spreads), no decomposition information."""
    A = nx.to_numpy_array(G)
    n = G.number_of_nodes()
    d = A.sum(axis=1)
    dinv = np.divide(1.0, np.sqrt(d), out=np.zeros_like(d), where=d > 0)
    Lnorm = np.eye(n) - (dinv[:, None] * A * dinv[None, :])
    ev = np.sort(eigvalsh(Lnorm))         # eigenvalues in [0, 2]

    L = np.diag(d) - A
    feats = {
        "nlap_min_nonzero": ev[1] if len(ev) > 1 else 0.0,
        "nlap_max": ev[-1],
        "nlap_mean": ev.mean(),
        "nlap_std": ev.std(),
        # spectral density near 0 and 2 (bipartite-ness signal)
        "nlap_frac_low": float((ev < 0.5).mean()),
        "nlap_frac_high": float((ev > 1.5).mean()),
    }
    # Heat-kernel trace tr(exp(-tL))/n: size-normalized diffusion signature, from
    # L's eigenvalues to avoid repeated matrix exponentials.
    lap_ev = np.sort(eigvalsh(L))
    for t in HEAT_TIMES:
        feats[f"heat_t{t}"] = float(np.exp(-t * lap_ev).sum() / n)
    return feats


def pauli_structure(g6: str) -> dict:
    """Features from BOTH decompositions: Pauli term count and Hamming-weight
    stats, plus the matching count. The label is cx_pauli - cx_matching, so these
    proxies mirror the label's structure and are borderline circular. Kept in
    their own group to report separately.

    Pauli side via Qiskit SparsePauliOp.from_operator (tensorized decomposition of
    Hantzko, Binkowski & Gupta, arXiv:2310.13421): same term count as term-by-term
    but far faster (validated on every persisted graph). Matching count still comes
    from the reference implementation.
    """
    import numpy as _np
    from qiskit.quantum_info import Operator, SparsePauliOp

    from ctqw_matching_decomp.core import MatchingDecomposition, MultiEdgeGraph
    from ctqw_matching_decomp.utils.graph.g6_utils import g6_to_edge_set

    G = MultiEdgeGraph(g6_to_edge_set(g6))

    spo = SparsePauliOp.from_operator(Operator(G.hamiltonian))
    keep = _np.abs(spo.coeffs) > _PAULI_ATOL
    nt = int(keep.sum())

    md = MatchingDecomposition(G, heuristic=_MATCH_HEURISTIC,
                               n_trials=_MATCH_N_TRIALS, seed=_MATCH_SEED)
    nm = md.num_matchings() if callable(md.num_matchings) else md.num_matchings

    feats = {
        "pauli_num_terms": float(nt),
        "num_matchings": float(nm),
        # Ratio and difference mirror label = cx_pauli - cx_matching: the winner
        # is set by relative term counts, not either count alone.
        "term_ratio": float(nt) / max(float(nm), 1.0),
        "term_diff": float(nt) - float(nm),
    }
    # Hamming weight of a Pauli string is the number of non-identity factors.
    labels = [str(p) for p, k in zip(spo.paulis, keep) if k]
    wts = _np.array([sum(1 for ch in lab if ch != "I") for lab in labels],
                    dtype=float)
    if wts.size:
        feats.update({"pauli_wt_mean": wts.mean(), "pauli_wt_max": wts.max(),
                      "pauli_wt_std": wts.std()})
    else:
        feats.update({"pauli_wt_mean": 0.0, "pauli_wt_max": 0.0, "pauli_wt_std": 0.0})
    return feats


