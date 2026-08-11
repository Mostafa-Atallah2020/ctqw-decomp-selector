"""Graph properties utilities for CTQW analysis.

Standalone NetworkX reimplementation of the ctqw-matching-decomp graph-property
helpers, with exact VF2 automorphism group size/orbit count and extra features
(degree stats, cycle/triangle counts, spectral gap, matching-decomp proxies).
The enumeration-based features (automorphism, clique) can hang on large dense
graphs and are gated by the with_automorphism / with_clique flags.
"""

from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------

def calculate_graph_properties(graph: nx.Graph,
                               with_automorphism: bool = True,
                               with_clique: bool = True) -> Dict:
    """Calculate graph features for one connected graph. Returns a dict with
    stable keys (int/float/bool values).

    Two features use worst-case-exponential enumerations that can hang on large
    dense graphs: with_automorphism=False omits automorphism_group_size and
    orbit_count. with_clique=False omits clique_number. Disable both for large
    graphs, on by default for small ones.
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    props: Dict = {}

    # size
    props["edge_count"] = m
    props["edge_density"] = nx.density(graph)
    props["is_bipartite"] = nx.is_bipartite(graph)

    # diameter (None if disconnected, connected input expected)
    props["diameter"] = nx.diameter(graph) if nx.is_connected(graph) else None

    # degrees
    degrees = [d for _, d in graph.degree()]
    props["max_degree"] = max(degrees) if degrees else 0
    props["min_degree"] = min(degrees) if degrees else 0
    props["avg_degree"] = sum(degrees) / n if n else 0.0
    props["degree_variance"] = float(np.var(degrees)) if degrees else 0.0
    props["is_regular"] = props["max_degree"] == props["min_degree"]

    # connectivity / cycles
    props["is_tree"] = (m == n - 1) and nx.is_connected(graph)
    # cycle space dimension = m - n + 1 for connected graphs
    props["cycle_count"] = m - n + 1 if nx.is_connected(graph) else None
    props["triangle_count"] = sum(nx.triangles(graph).values()) // 3

    # clique number: maximal-clique enumeration is exponential on dense graphs
    # (see with_clique).
    if with_clique:
        try:
            props["clique_number"] = max(len(c) for c in nx.find_cliques(graph))
        except ValueError:
            props["clique_number"] = 1

    # clustering coefficient
    props["avg_clustering"] = nx.average_clustering(graph)

    # spectral gap = algebraic connectivity (lambda_2 of Laplacian)
    props["spectral_gap"] = _laplacian_spectral_gap(graph)

    # matching-decomp proxies
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    props["max_matching_size"] = len(matching)
    # Vizing: chromatic index is max_degree or max_degree+1, so max_degree lower-
    # bounds the matchings needed to decompose the edge set.
    props["chromatic_index_lower_bound"] = props["max_degree"]

    # exact automorphism group size + orbit count
    if with_automorphism:
        group_size, orbit_count = _automorphism_via_networkx(graph)
        props["automorphism_group_size"] = group_size
        props["orbit_count"] = orbit_count

    return props


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _laplacian_spectral_gap(graph: nx.Graph) -> float:
    """Algebraic connectivity = second-smallest Laplacian eigenvalue (0 for
    disconnected or n < 2)."""
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0
    L = nx.laplacian_matrix(graph).toarray().astype(float)
    eigvals = np.linalg.eigvalsh(L)
    # eigvalsh sorts ascending, lambda_1 ~ 0 for connected
    return float(eigvals[1])


def _automorphism_via_networkx(graph: nx.Graph) -> tuple[int, int]:
    """Return (|Aut(G)|, orbit_count) via networkx's VF2 engine.

    |Aut(G)| = self-isomorphism count from GraphMatcher. Orbits = union-find
    components over the (i, sigma(i)) relation across all automorphisms. Pure
    Python and portable, can hang on large dense graphs (the g6 extractor
    disables it, see dataset.features).
    """
    from networkx.algorithms import isomorphism

    n = graph.number_of_nodes()
    if n == 0:
        return 1, 0
    if n == 1:
        return 1, 1

    gm = isomorphism.GraphMatcher(graph, graph)

    # Union-find over node indices to collect orbits.
    parent = {v: v for v in graph.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    count = 0
    for mapping in gm.isomorphisms_iter():
        count += 1
        for src, dst in mapping.items():
            union(src, dst)

    if count == 0:
        # Unreachable (identity is always an automorphism), safety fallback.
        return 1, n

    orbits = {find(v) for v in graph.nodes()}
    return count, len(orbits)


# ---------------------------------------------------------------------------
# small helpers retained for backwards compatibility
# ---------------------------------------------------------------------------

def hamming_distance(s1: str, s2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def compute_hamming_statistics(edges: set) -> Dict:
    """Compute Hamming distance statistics for a set of bitstring edges."""
    if not edges:
        return {"total_hamming": 0, "avg_hamming": 0.0,
                "hamming_1_count": 0, "hamming_gt1_count": 0,
                "min_hamming": 0, "max_hamming": 0}

    dists = [sum(a != b for a, b in zip(u, v)) for u, v in edges]
    h1 = sum(1 for d in dists if d == 1)
    return {
        "total_hamming": sum(dists),
        "avg_hamming": float(np.mean(dists)),
        "min_hamming": int(min(dists)),
        "max_hamming": int(max(dists)),
        "hamming_1_count": h1,
        "hamming_gt1_count": len(dists) - h1,
    }
