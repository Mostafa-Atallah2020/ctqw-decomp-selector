"""Build a balanced balanced training set by selecting graph_ids from both datasets.

Erdos-Renyi is ~99.8% Pauli-wins and structured is majority matching-wins.
Neither alone is a good training set (see the dataset analysis). This assembles a
class-balanced balanced by sampling graph_ids so that, WITHIN EACH vertex count,
matching-wins and Pauli-wins are equally represented. Balancing within n (not
just overall) prevents a size/dataset shortcut: the model cannot learn "large =>
matching, small => Pauli" because at every n it sees both outcomes.

Matching-wins are the scarce class (mostly from structured). Pauli-wins are
abundant (mostly from ER). For each n we take min(#matching, #pauli) of each.

Outputs data/balanced/{features,labels}/ with the same schema as the source
datasets, plus a `source_dataset` column, so downstream code is unchanged. Ties
(delta_cx == 0) are excluded.

This module is the importable library. Run it via scripts/build_balanced.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = _REPO_ROOT / "data"
DEFAULT_OUT = _REPO_ROOT / "data" / "balanced"


log = logging.getLogger("balanced")


def _load(data_dir: Path, dataset: str):
    """Load labels joined to features for one dataset, tagged with its source."""
    import pandas as pd
    feat = pd.read_csv(data_dir / dataset / "features" / "graph_features.csv")
    lab = pd.read_csv(data_dir / dataset / "labels" / "labels.csv")
    df = lab.merge(feat, on="graph_id", how="inner", suffixes=("", "_feat"))
    # n_vertices exists in both, keep one clean copy.
    if "n_vertices_feat" in df.columns:
        df = df.drop(columns=["n_vertices_feat"])
    df["source_dataset"] = dataset
    return df


def _load_unlabeled(data_dir: Path, dataset: str):
    """Featurized graphs with NO label row (an anti-join against labels.csv).

    These are the graphs the pipeline generated and featurized but could not
    afford to label. Labeling cost grows as ~n^3.2, so it is ~0.5 h/graph at
    n=512 and ~4.7 h/graph at n=1024. They carry every feature and a real
    graph_id, so they stay traceable and could be labeled later.
    """
    import pandas as pd
    feat = pd.read_csv(data_dir / dataset / "features" / "graph_features.csv")
    lab = pd.read_csv(data_dir / dataset / "labels" / "labels.csv")
    df = feat[~feat.graph_id.isin(set(lab.graph_id))].copy()
    df["source_dataset"] = dataset
    return df


def build(data_dir: Path, out_dir: Path, seed: int = 0) -> dict:
    """Select a within-n class-balanced balanced and write it to out_dir.

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
                              m.source_dataset.value_counts().items()},
            "pauli_from": {k: int(v) for k, v in
                           p.source_dataset.value_counts().items()},
        }

    balanced = pd.concat(picks, ignore_index=True) if picks else both.iloc[:0]

    # Split back into the same two-file layout as the source datasets.
    feat_cols = [c for c in balanced.columns if c in _feature_columns(data_dir)]
    label_cols = ["graph_id", "n_vertices", "cx_matching", "cx_pauli",
                  "depth_matching", "depth_pauli", "delta_cx", "delta_depth",
                  "labeling_time"]
    label_cols = [c for c in label_cols if c in balanced.columns]

    unlabeled = pd.concat(
        [_load_unlabeled(data_dir, c) for c in ("er", "structured")],
        ignore_index=True)
    features_out = pd.concat(
        [balanced[["source_dataset"] + feat_cols],
         unlabeled[["source_dataset"] + feat_cols]],
        ignore_index=True)

    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    features_out.to_csv(out_dir / "features" / "graph_features.csv", index=False)
    balanced[["source_dataset"] + label_cols].to_csv(
        out_dir / "labels" / "labels.csv", index=False)

    matching_total = int((balanced.delta_cx > 0).sum())
    pauli_total = int((balanced.delta_cx < 0).sum())
    manifest = {
        "source": "balanced_within_n",
        "seed": seed,
        "total": len(balanced),
        "unlabeled_carried": {
            "total": int(len(unlabeled)),
            "per_size": {int(k): int(v) for k, v in
                         unlabeled.n_vertices.value_counts().sort_index().items()},
            "per_dataset": {k: int(v) for k, v in
                           unlabeled.source_dataset.value_counts().items()},
        },
        "matching_wins": matching_total,
        "pauli_wins": pauli_total,
        "balance": "50/50 within each vertex count",
        "per_n": per_n,
        "source_mix": {k: int(v) for k, v in
                       balanced.source_dataset.value_counts().items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("balanced: %d graphs (%d matching / %d pauli) -> %s",
             len(balanced), matching_total, pauli_total, out_dir)
    return manifest


def _feature_columns(data_dir: Path) -> set:
    """Feature-CSV columns (from the ER dataset header), minus graph_id."""
    header = (data_dir / "er" / "features" / "graph_features.csv"
              ).read_text().splitlines()[0].split(",")
    return {c for c in header if c != "graph_id"} | {"graph_id"}
