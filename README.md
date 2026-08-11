Code repository for the paper:

> **Predicting Resource Efficient Hamiltonian Decomposition for Continuous-Time Quantum Walk Simulations**
>
> Mostafa Atallah, Rebekah Herrman, Zain H. Saleem

## Overview

Given a graph, predict whether the **matching** or the **Pauli** decomposition produces the smaller CX gate count when simulating a continuous-time quantum walk (CTQW) Hamiltonian, so a compiler can choose per graph rather than always defaulting to one.

- **Label**: `y = 1[delta_cx > 0]` where `delta_cx = CX_Pauli - CX_matching`. Positive means matching is cheaper.
- **Model**: a small neural network (one 32-node tanh hidden layer) on twelve features, ten topological graph properties plus the two decomposition counts (number of Pauli terms and number of matchings).
- **Result**: MCC 0.593 on the complete 8-vertex population, transferring to larger unseen sizes (MCC 0.785 at N=8 to 1.0 at N >= 64).

Ground-truth CX counts come from [`ctqw-matching-decomp`](https://github.com/Mostafa-Atallah2020/ctqw-matching-decomp).

## Installation

```bash
git clone https://github.com/Mostafa-Atallah2020/ctqw-decomp-selector.git
cd ctqw-decomp-selector
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datasets

| dataset | graphs | description |
|---|---|---|
| `mckay` | 11,117 | Complete population of connected 8-vertex graphs (nauty `geng`). Primary. Roughly 25 to 1 imbalance. |
| `er` | 10,989 | Erdos-Renyi sweep. Pauli almost always cheaper. |
| `structured` | 9,092 | Counting-path variants. Matching cheaper on the majority. |
| `balanced` | 13,442 | Within-size class-balanced held-out set from er and structured. |

Each lives under `data/<dataset>/{g6,features,labels}/`, keyed by graph6 hash.

## Usage

### Build a dataset

```bash
python scripts/dataset.py --stage generate --dataset er     # graphs
python scripts/dataset.py --stage features --dataset er     # topological features
python scripts/dataset.py --stage label    --dataset er     # CX-count labels (slow, resumable)
```

Add `--limit 100` to smoke-test any stage. The mckay population is enumerated with nauty `geng`, and its spectral and decomposition features come from `python scripts/dataset.py --stage persist --dataset mckay`.

### Train and evaluate

```bash
python scripts/evaluate_models.py --experiment benchmark    # compare twelve classifiers
python scripts/evaluate_models.py --experiment train        # train the shipped model
python scripts/evaluate_models.py --experiment per-size     # per-size transfer
python scripts/evaluate_models.py --experiment lofo         # leave-one-family-out transfer
python scripts/evaluate_features.py --mode allmetrics       # feature-set experiments
```

### Figures

```bash
python scripts/plot_figures.py --fig all
```

## Project Structure

```
ctqw-decomp-selector/
├── scripts/
│   ├── dataset.py               # build + analyze (generate|label|features|persist|balanced|analyze)
│   ├── evaluate_features.py     # feature-set experiments (features|allmetrics|spectral)
│   ├── evaluate_models.py       # train|tune|benchmark|ablation|overlap|lofo|per-size
│   ├── plot_figures.py          # McKay figures (all|analysis|importance|roc-pr)
│   └── predict_unlabeled.py     # predict on unlabeled graphs, label-free scoring
├── src/
│   ├── dataset/                 # generate.py, features.py, label.py, balanced.py
│   ├── model/                   # evaluation.py (shared lib), logistic.py, plots.py
│   ├── analysis/                # dataset_analysis.py
│   ├── graph/                   # properties.py
│   └── progress.py
├── data/                        # er, structured, mckay, balanced (g6, features, labels)
└── results/                     # analysis, evaluation, predictions
```

## Citation

```bibtex
@article{atallah2026ctqwdecomp,
  title={Predicting Resource Efficient Hamiltonian Decomposition for Continuous-Time Quantum Walk Simulations},
  author={Atallah, Mostafa and Herrman, Rebekah and Saleem, Zain H.},
  journal={arXiv preprint arXiv:TBA},
  year={2026}
}
```

## Contact

For questions or issues, please contact: matalla3@vols.utk.edu
