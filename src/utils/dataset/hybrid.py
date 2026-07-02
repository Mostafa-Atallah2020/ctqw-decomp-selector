"""Build a balanced hybrid training set by selecting graph_ids from both corpora.

Erdos-Renyi is ~99.8% Pauli-wins and structured is majority matching-wins;
neither alone is a good training set (see the corpus analysis). This assembles a
class-balanced hybrid by sampling graph_ids so that, WITHIN EACH vertex count,
matching-wins and Pauli-wins are equally represented. Balancing within n (not
just overall) prevents a size/corpus shortcut: the model cannot learn "large =>
matching, small => Pauli" because at every n it sees both outcomes.

Matching-wins are the scarce class (mostly from structured); Pauli-wins are
abundant (mostly from ER). For each n we take min(#matching, #pauli) of each.

Outputs data/hybrid/{features,labels}/ with the same schema as the source
corpora, plus a `source_corpus` column, so downstream code is unchanged. Ties
(delta_cx == 0) are excluded.

This module is the importable library; run it via scripts/build_hybrid.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = _REPO_ROOT / "data"
DEFAULT_OUT = _REPO_ROOT / "data" / "hybrid"


log = logging.getLogger("hybrid")


def _load(data_dir: Path, corpus: str):
    """Load labels joined to features for one corpus, tagged with its source."""
    import pandas as pd
    feat = pd.read_csv(data_dir / corpus / "features" / "graph_features.csv")
    lab = pd.read_csv(data_dir / corpus / "labels" / "labels.csv")
    df = lab.merge(feat, on="graph_id", how="inner", suffixes=("", "_feat"))
    # n_vertices exists in both; keep one clean copy.
    if "n_vertices_feat" in df.columns:
        df = df.drop(columns=["n_vertices_feat"])
    df["source_corpus"] = corpus
    return df


def build(data_dir: Path, out_dir: Path, seed: int = 0) -> dict:
    """Select a within-n class-balanced hybrid and write it to out_dir.

    Returns a manifest dict describing the per-n composition.
    """
    import numpy as np
    import pandas as pd

    both = pd.concat([_load(data_dir, "er"), _load(data_dir, "structured")],
                     ignore_index=True)
    rng = np.random.RandomState(seed)

    picks = []
    per_n = {}
    for n, g in both.groupby("n_vertices"):
        matching = g[g.delta_cx > 0]
        pauli = g[g.delta_cx < 0]
        take = min(len(matching), len(pauli))
        if take == 0:
            per_n[int(n)] = {"take_per_class": 0, "total": 0,
                             "matching_avail": len(matching),
                             "pauli_avail": len(pauli)}
            continue
        m = matching.sample(n=take, random_state=rng)
        p = pauli.sample(n=take, random_state=rng)
        picks.append(pd.concat([m, p]))
        per_n[int(n)] = {
            "take_per_class": int(take), "total": int(2 * take),
            "matching_avail": int(len(matching)),
            "pauli_avail": int(len(pauli)),
            "matching_from": {k: int(v) for k, v in
                              m.source_corpus.value_counts().items()},
            "pauli_from": {k: int(v) for k, v in
                           p.source_corpus.value_counts().items()},
        }

    hybrid = pd.concat(picks, ignore_index=True) if picks else both.iloc[:0]

    # Split back into the same two-file layout as the source corpora.
    feat_cols = [c for c in hybrid.columns if c in _feature_columns(data_dir)]
    label_cols = ["graph_id", "n_vertices", "cx_matching", "cx_pauli",
                  "depth_matching", "depth_pauli", "delta_cx", "delta_depth",
                  "labeling_time"]
    label_cols = [c for c in label_cols if c in hybrid.columns]

    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    hybrid[["source_corpus"] + feat_cols].to_csv(
        out_dir / "features" / "graph_features.csv", index=False)
    hybrid[["source_corpus"] + label_cols].to_csv(
        out_dir / "labels" / "labels.csv", index=False)

    matching_total = int((hybrid.delta_cx > 0).sum())
    pauli_total = int((hybrid.delta_cx < 0).sum())
    manifest = {
        "source": "hybrid_within_n_balanced",
        "seed": seed,
        "total": len(hybrid),
        "matching_wins": matching_total,
        "pauli_wins": pauli_total,
        "balance": "50/50 within each vertex count",
        "per_n": per_n,
        "source_mix": {k: int(v) for k, v in
                       hybrid.source_corpus.value_counts().items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("hybrid: %d graphs (%d matching / %d pauli) -> %s",
             len(hybrid), matching_total, pauli_total, out_dir)
    return manifest


def _feature_columns(data_dir: Path) -> set:
    """Feature-CSV columns (from the ER corpus header), minus graph_id."""
    header = (data_dir / "er" / "features" / "graph_features.csv"
              ).read_text().splitlines()[0].split(",")
    return {c for c in header if c != "graph_id"} | {"graph_id"}
