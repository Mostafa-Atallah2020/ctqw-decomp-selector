"""Graph properties utilities for CTQW analysis.

A standalone reimplementation inspired by the graph-property helpers in the
ctqw-matching-decomp repo (https://github.com/Mostafa-Atallah2020/ctqw-matching-decomp). 
All values are computed here directly via NetworkX.
It differs from that reference in two ways:

- Exact automorphism group size and orbit count (NetworkX VF2), replacing the
  reference's rough heuristic estimator.
- Extra features: min_degree, degree_variance, is_regular, is_tree,
  cycle_count, triangle_count, spectral_gap, and the matching-decomposition
  proxies max_matching_size and chromatic_index_lower_bound.

The two enumeration-based features (automorphism, clique) can hang on large
dense graphs and are gated by the with_automorphism / with_clique flags.
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
    """Calculate graph features for one graph. Assumes graph is connected.

    Returns a dict with stable keys; values are int / float / bool.

    Two features rely on enumerations that are exponential in the worst case
    and can hang on large dense graphs:
      - with_automorphism=False omits automorphism_group_size and orbit_count
        (VF2 self-isomorphism enumeration).
      - with_clique=False omits clique_number (maximal-clique enumeration).
    Callers working with large graphs (e.g. the synthetic n up to 128 corpus)
    should disable these; they stay on by default for small graphs.
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    props: Dict = {}

    # size
    props["edge_count"] = m
    props["edge_density"] = nx.density(graph)
    props["is_bipartite"] = nx.is_bipartite(graph)

    # diameter (None only if disconnected; we expect connected input)
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
    # for connected graphs, cycle space dimension = m - n + 1
    props["cycle_count"] = m - n + 1 if nx.is_connected(graph) else None
    props["triangle_count"] = sum(nx.triangles(graph).values()) // 3

    # clique number (max clique size); for organics typically 2 or 3.
    # Maximal-clique enumeration is exponential on large dense graphs, so it is
    # optional (see with_clique).
    if with_clique:
        try:
            props["clique_number"] = max(len(c) for c in nx.find_cliques(graph))
        except ValueError:
            props["clique_number"] = 1

    # clustering coefficient
    props["avg_clustering"] = nx.average_clustering(graph)

    # spectral gap = lambda_2 - lambda_1 of Laplacian (algebraic connectivity)
    props["spectral_gap"] = _laplacian_spectral_gap(graph)

    # matching-decomp proxies
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    props["max_matching_size"] = len(matching)
    # Vizing's lower bound for chromatic index: max_degree.
    # The chromatic index of a graph is max_degree or max_degree + 1.
    # max_degree is also a lower bound on the number of matchings needed
    # to decompose the edge set.
    props["chromatic_index_lower_bound"] = props["max_degree"]

    # exact automorphism group size + orbit count via networkx
    if with_automorphism:
        group_size, orbit_count = _automorphism_via_networkx(graph)
        props["automorphism_group_size"] = group_size
        props["orbit_count"] = orbit_count

    return props


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _laplacian_spectral_gap(graph: nx.Graph) -> float:
    """Algebraic connectivity = second-smallest Laplacian eigenvalue.

    For a connected graph, lambda_1 = 0 and lambda_2 > 0; the gap lambda_2 - 0
    is the algebraic connectivity. For disconnected graphs returns 0.
    """
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0
    L = nx.laplacian_matrix(graph).toarray().astype(float)
    eigvals = np.linalg.eigvalsh(L)
    # eigvalsh returns sorted ascending; lambda_1 is ~0 for connected
    return float(eigvals[1])


def _automorphism_via_networkx(graph: nx.Graph) -> tuple[int, int]:
    """Return (|Aut(G)|, orbit_count) using networkx's VF2 isomorphism engine.

    Approach:
      - |Aut(G)| = number of self-isomorphisms, enumerated by GraphMatcher.
      - Orbits = connected components in the union-find over the (i, sigma(i))
        relation across all automorphisms.

    Pure Python and slower than nauty, but portable. For small graphs
    (n <= 30) this runs in milliseconds; it can hang on large dense graphs,
    so the g6 feature extractor disables it (see dataset.features).
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
        # Should never happen (identity is always an automorphism), but be safe.
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
