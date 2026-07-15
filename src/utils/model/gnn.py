"""Graph neural network baseline operating on the adjacency structure.

A separate model is trained and evaluated per corpus (both er and structured by
default), so that the two corpora are compared under an identical protocol.

The architecture is a Graph Isomorphism Network (GIN; Xu et al., ICLR 2019), the
maximally expressive message-passing architecture under the 1-Weisfeiler-Leman
discrimination bound. Node inputs are restricted to degree (raw and normalized); all
further structural features are learned. The graph-level readout concatenates
sum-pooling (size-sensitive) and mean-pooling (size-invariant), leaving the classifier
free to use, or ignore, graph size.

The objective is class-weighted cross-entropy, which the class imbalance requires:
on the Erdos-Renyi corpus, the more extreme of the two, the positive class constitutes
only 0.2% of examples, for which an unweighted objective is minimized by the constant
Pauli prediction.

Results are assessed by CX savings rather than MCC. Under severe class imbalance a
classifier can increase MCC by predicting the minority class more frequently while
reducing realized savings, since each false positive on a Pauli-favorable graph incurs
that graph's full cost gap. A model attaining MCC 0.1 with negative CX savings is
therefore inferior to the constant-Pauli baseline, which attains MCC 0 and zero
savings.

Command-line interface: scripts/benchmark_gnn.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.data import Data  # type: ignore[import-not-found]
from torch_geometric.loader import DataLoader  # type: ignore[import-not-found]
from torch_geometric.nn import (  # type: ignore[import-not-found]
    GINConv, global_add_pool, global_mean_pool)

from utils.analysis.corpus_analysis import DEFAULT_DATA
from utils.progress import Progress

log = logging.getLogger("gnn")

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _ROOT / "results" / "gnn_model"

CORPORA = ["er", "structured"]
NMAX = 256             # the largest labeled size
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60
BATCH_SIZE = 64
LOG_EVERY = 10         # print train/val metrics every N epochs (0 disables)

METRICS = ["accuracy", "precision", "recall", "specificity", "npv", "f1", "mcc",
           "cohen_kappa", "roc_auc", "pr_auc", "log_loss", "savings"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_graphs(corpus: str, data_dir: Path = DEFAULT_DATA,
                prog: Progress | None = None,
                nmax: int = NMAX) -> tuple[list, pd.DataFrame]:
    """Every labeled graph (n <= nmax) as a PyG Data object, with its adjacency
    reconstructed.

    data/<corpus>/g6/all.g6 holds one graph6 string per line, in the same order as
    the rows of graph_features.csv, so the two join positionally. graph6 is lossless:
    the full edge set comes back. That edge set is exactly what the feature CSV
    discards, and exactly what this experiment exists to use.
    """
    d = data_dir / corpus
    g6 = (d / "g6" / "all.g6").read_text().split()
    feat = pd.read_csv(d / "features" / "graph_features.csv").reset_index(drop=True)
    lab = pd.read_csv(d / "labels" / "labels.csv")

    feat["g6"] = g6[: len(feat)]
    df = feat.merge(lab[["graph_id", "delta_cx"]], on="graph_id", how="inner")
    df = df[(df.n_vertices <= nmax) & (df.delta_cx != 0)].copy()
    df["y"] = (df.delta_cx > 0).astype(int)

    if prog is not None:
        prog.total = (prog.total or 0) + len(df)

    data = []
    for _, r in df.iterrows():
        G = nx.from_graph6_bytes(r.g6.encode())
        n = G.number_of_nodes()
        A = nx.to_scipy_sparse_array(G, format="coo")

        deg = np.array([k for _, k in G.degree()], dtype=np.float32)
        x = torch.tensor(np.stack([deg, deg / max(n - 1, 1)], axis=1))

        data.append(Data(
            x=x,
            edge_index=torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long),
            y=torch.tensor([int(r.y)], dtype=torch.long),
            dcx=torch.tensor([float(r.delta_cx)]),
            nv=torch.tensor([int(r.n_vertices)]),
        ))
        if prog is not None:
            prog.step(f"{corpus}: reading adjacency", key=corpus)

    return data, df


def split(data: list, df: pd.DataFrame, seed: int = 0):
    """Leakage-safe 80/20 split, stratified by (size, class) so every test set spans
    every vertex count."""
    key = df.n_vertices.astype(str) + "_" + df.y.astype(str)
    tr, te = train_test_split(np.arange(len(data)), test_size=0.2,
                              random_state=seed, stratify=key)
    return [data[i] for i in tr], [data[i] for i in te]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class GIN(torch.nn.Module):
    """Graph Isomorphism Network: message-passing layers followed by a graph readout.

    The readout concatenates sum- and mean-pooling. Sum-pooling scales with graph size
    while mean-pooling does not, so the classifier may either use size as a feature or
    disregard it. This is relevant because both corpora are strongly size-confounded
    (ER is entirely Pauli for n>=16, structured entirely matching for n>=64); the
    concatenated readout neither supplies size as a free predictor nor withholds it.
    """

    def __init__(self, in_dim: int = 2, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(layers):
            d = in_dim if i == 0 else hidden
            self.convs.append(GINConv(Sequential(
                Linear(d, hidden), BatchNorm1d(hidden), ReLU(),
                Linear(hidden, hidden), ReLU(),
            )))
        self.head = Sequential(Linear(hidden * 2, hidden), ReLU(), Linear(hidden, 2))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
        g = torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        return self.head(g)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred, dcx, score) -> dict:
    """The same twelve metrics reported in results/logistic_model/metrics.csv, so these rows
    read alongside the twelve-model ladder."""
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) > 1

    dcx = np.asarray(dcx, dtype=float)
    realizable = dcx[dcx > 0].sum()
    savings = (float(dcx[np.asarray(y_pred) == 1].sum() / realizable)
               if realizable > 0 else float("nan"))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if both else 0.0,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, score) if both else float("nan"),
        "pr_auc": average_precision_score(y_true, score) if both else float("nan"),
        "log_loss": log_loss(y_true, score, labels=[0, 1]) if both else float("nan"),
        "savings": savings,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _predict(model, graphs: list) -> np.ndarray:
    """P(matching) for every graph, from the softmax head."""
    model.eval()
    probs = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=256):
            probs.append(F.softmax(model(b.x, b.edge_index, b.batch), dim=1)[:, 1])
    return torch.cat(probs).numpy()


def fit_and_score(train: list, test: list, seed: int, epochs: int,
                  prog: Progress | None = None, corpus: str = "",
                  history: list | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Train one GIN and score it on the held-out graphs.

    Appends per-epoch train/validation log-loss to `history` (if given) in the schema
    results/logistic_model/training_history.csv uses, so the same plotting code renders the GIN
    convergence curve. Unlike logistic regression, whose ten lbfgs iterations converge
    almost immediately, this is real gradient descent and the curve is informative.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ytr = np.array([int(d.y) for d in train])
    yte = np.array([int(d.y) for d in test])
    dcx = np.array([float(d.dcx) for d in test])

    model = GIN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_pos = max(int((ytr == 1).sum()), 1)
    weight = torch.tensor([1.0, max((ytr == 0).sum() / n_pos, 1.0)], dtype=torch.float)

    loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    for ep in range(1, epochs + 1):
        model.train()
        for b in loader:
            opt.zero_grad()
            F.cross_entropy(model(b.x, b.edge_index, b.batch), b.y,
                            weight=weight).backward()
            opt.step()

        # Evaluate for the convergence curve (recorded on the FIRST seed only: five
        # overlapping curves would be noise) and for the live per-epoch print. Only
        # score when something actually consumes it, since _predict runs the whole
        # train + test set through the model.
        want_curve = history is not None and seed == 0
        want_print = LOG_EVERY and (ep % LOG_EVERY == 0 or ep == epochs)
        if want_curve or want_print:
            p_tr = _predict(model, train)
            p_te = _predict(model, test)
            tr_loss = log_loss(ytr, p_tr, labels=[0, 1])
            va_loss = log_loss(yte, p_te, labels=[0, 1])

            if want_curve:
                history.append({"corpus": corpus, "iteration": ep,
                                "train_loss": tr_loss, "val_loss": va_loss})
            if want_print:
                pred = (p_te >= 0.5).astype(int)
                va_mcc = (matthews_corrcoef(yte, pred)
                          if len(np.unique(yte)) > 1 else 0.0)
                va_rec = recall_score(yte, pred, zero_division=0)
                log.info("  %-9s seed %d  epoch %3d/%d  "
                         "train_loss %.4f  val_loss %.4f  val_MCC %.3f  val_recall %.3f",
                         corpus, seed, ep, epochs, tr_loss, va_loss, va_mcc, va_rec)

        if prog is not None:
            prog.step(f"{corpus} seed={seed} epoch {ep}/{epochs}", key=corpus)

    p = _predict(model, test)
    pred = (p >= 0.5).astype(int)
    return metrics(yte, pred, dcx, p), pred, p


# ---------------------------------------------------------------------------
def run(data_dir: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT,
        corpora: list[str] | None = None, seeds: list[int] | None = None,
        epochs: int = EPOCHS, nmax: int = NMAX) -> dict:
    """Train a GIN per corpus and write results/gnn_model/metrics{,_by_size}.csv."""
    corpora = corpora or CORPORA
    seeds = seeds or SEEDS
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: rebuild every adjacency matrix. One Progress with a per-corpus key, so
    # the ETA stays sane even though ER's dense n=256 graphs cost far more than
    # structured's sparse ones.
    build = Progress(unit="graph")
    loaded = {}
    for corpus in corpora:
        loaded[corpus] = load_graphs(corpus, data_dir, build, nmax=nmax)
    build.done()

    # Stage 2: train. Every (corpus, seed, epoch) triple is one step, so the ETA is
    # meaningful from the first epoch rather than only after the first seed.
    fit = Progress(total=sum(len(seeds) * epochs for _ in corpora), unit="epoch")

    per_seed_rows, rows, per_size_rows, summary = [], [], [], []
    history, predictions = [], []          # feed the shared plotting code
    for corpus in corpora:
        data, df = loaded[corpus]
        train, test = split(data, df)
        yte = np.array([int(d.y) for d in test])
        nv = np.array([int(d.nv) for d in test])

        log.info("%s: %d graphs, %d matching-wins (%.2f%%); train %d / test %d",
                 corpus, len(data), int(df.y.sum()), 100 * df.y.mean(),
                 len(train), len(test))

        runs, first_pred = [], None
        for seed in seeds:
            m, pred, proba = fit_and_score(train, test, seed, epochs, fit, corpus,
                                           history)
            runs.append(m)
            per_seed_rows.append({"corpus": corpus, "seed": seed,
                                  **{k: m[k] for k in METRICS}})
            if first_pred is None:
                first_pred = pred
                predictions.extend({
                    "corpus": corpus,
                    "n_vertices": int(n),
                    "y_true": int(y),
                    "proba": float(p),
                } for n, y, p in zip(nv, yte, proba))

            # Flush after every seed so an interrupted run stays readable.
            _flush(out_dir, per_seed_rows, history, predictions)

        agg = {"corpus": corpus, "n_train": len(train), "n_test": len(test),
               "n_matching_wins": int(yte.sum())}
        for metric in METRICS:
            vals = np.array([r[metric] for r in runs], dtype=float)
            if np.isnan(vals).all():
                agg[f"{metric}_mean"] = agg[f"{metric}_std"] = float("nan")
            else:
                agg[f"{metric}_mean"] = np.nanmean(vals)
                agg[f"{metric}_std"] = np.nanstd(vals)
        rows.append(agg)
        summary.append(agg)

        # Within-size MCC. A pooled score can be inflated by the model learning
        # "large => matching" from the size distribution, so score each size alone.
        for n in sorted(set(nv)):
            m = nv == n
            single = len(set(yte[m])) < 2
            per_size_rows.append({
                "corpus": corpus,
                "n_vertices": int(n),
                "n_graphs": int(m.sum()),
                "matching_wins": int(yte[m].sum()),
                "mcc": (float("nan") if single
                        else matthews_corrcoef(yte[m], first_pred[m])),
                "single_class": single,
            })
        # metrics.csv / metrics_by_size.csv share the logistic model's filenames but
        # live in results/gnn_model/, so there is no collision. Flushed per corpus.
        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
        pd.DataFrame(per_size_rows).to_csv(out_dir / "metrics_by_size.csv",
                                           index=False)
    fit.done()

    log.info("wrote metrics.csv, metrics_by_size.csv, per_seed.csv, "
             "training_history.csv and test_predictions.csv to %s", out_dir)

    return {"summary": summary, "by_size": per_size_rows, "out_dir": out_dir,
            "epochs": epochs, "seeds": seeds}


def _flush(out_dir: Path, per_seed_rows: list, history: list,
           predictions: list) -> None:
    """Write the per-seed metrics, training curve and predictions gathered so far.

    Called after each seed so a long or interrupted run leaves partial results on
    disk. Cheap: these tables are small and rewritten whole each time.
    """
    pd.DataFrame(per_seed_rows).to_csv(out_dir / "per_seed.csv", index=False)
    if history:
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    if predictions:
        pd.DataFrame(predictions).to_csv(out_dir / "test_predictions.csv",
                                         index=False)


def plot_all(out_dir: Path = DEFAULT_OUT) -> None:
    """Render the GIN's diagnostic plots, reusing the logistic model's plotting code.

    Four of the six carry over. Two do not, and the reason is structural rather than
    an omission:

      coefficients        a GIN has no per-feature weight to plot. It has ~40k
                          parameters over learned node embeddings, and no single
                          number says "diameter pushes toward matching".
      decision_boundaries needs a boundary in FEATURE space. The GIN has no feature
                          space; it operates on the adjacency directly. That is the
                          entire point of the experiment.

    The absence of these two is itself the finding: the GIN buys structural access at
    the cost of the interpretability that made the logistic model shippable.

    Because the outputs live in their own directory (results/gnn_model/), they use the
    same filenames as results/logistic_model/, and the shared plotting code needs no renaming.
    """
    from utils.model.plots import (plot_calibration, plot_confusion,
                                   plot_roc_pr, plot_training_curve)

    hist = out_dir / "training_history.csv"
    preds = out_dir / "test_predictions.csv"

    if hist.exists():
        plot_training_curve(hist, out_dir)
    if preds.exists():
        plot_roc_pr(preds, out_dir)
        plot_confusion(preds, out_dir)
        plot_calibration(preds, out_dir)

    log.info("wrote GIN plots (training curve, ROC/PR, confusion, calibration) "
             "to %s", out_dir)
