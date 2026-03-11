# TGAN-OOD: Temporal GAN for Out-of-Distribution Social Event Detection

## Overview

**TGAN-OOD** is a graph-based social event detection framework for Twitter/X streams that explicitly handles **out-of-distribution (OOD)** scenarios — situations where new, previously unseen event types emerge over time, causing distribution shifts that degrade conventional models.

The key technical contributions are:

- **Temporal graph construction**: Edge weights are blended between semantic similarity and temporal proximity using parameters ρ (time balance), r (decay rate), σ (scaling), and ξ (graph density).
- **Graph Neural Network encoder**: A 2-layer Graph Attention Network (GAT) encodes message nodes into compact event-cluster embeddings.
- **GAN-based data augmentation**: A Generator produces synthetic SBERT-like embeddings to bridge the gap between training and test distributions; a Discriminator enforces realism.
- **JSD regularisation**: A differentiable Jensen-Shannon Divergence loss penalises the distribution gap between augmented training data and the test block, controlled by λ_JSD.

---

## Architecture

```
SBERT Embeddings (384-D)
        │
        ▼
Temporal Graph Builder          ← ρ, r, σ, ξ hyperparameters
(attribute + kNN edges)
        │
        ▼
GNN Encoder (2-layer GAT)       ← in_dim=384, hidden=128, out=128
        │
        ├──► ClusterHead        → Clustering Loss (cross-entropy)
        │
        ▼
Generator  ←── Noise (128-D)    ← φ controls sample proportion
        │
        ▼
Discriminator                   → Adversarial Loss (BCE)
        │
        ▼
JSD Loss (augmented vs. test)   ← λ_JSD controls regularisation weight
        │
        ▼
Total Loss = L_cluster + L_adv + λ_JSD × L_jsd
```

---

## Datasets

Download the Event2012 and Event2018 datasets from:

[Google Drive Link](https://drive.google.com/drive/folders/1zE-seeNPRFCo5L9wUG62i4KXKG-sNJ6e?usp=sharing)

Place the raw data under:
```
raw_data/
├── Event2012/
│   ├── 68841_tweets_multiclasses_filtered_0722_part1.npy
│   ├── 68841_tweets_multiclasses_filtered_0722_part2.npy
│   └── masks/test_mask.pt
└── Event2018/
    ├── french_tweets.npy
    └── data_splits/test_indices.pkl
```

---

## Installation

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install PyTorch Geometric (follow official guide for your OS/CUDA)
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch_geometric torch-scatter torch-sparse torch-cluster

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Preprocess datasets

```python
from tganood.preprocessor import preprocess_event2012, preprocess_event2018

preprocess_event2012()   # generates SBERT embeddings for Event2012
preprocess_event2018()   # generates SBERT embeddings for Event2018
```

### Step 2 — Run experiments

```bash
python -m tganood.run
```

Or call individual experiment functions:

```python
from tganood.run import (
    run_tganood_event2012_open_set,
    run_tganood_event2012_closed_set,
    run_tganood_event2018_open_set,
    run_tganood_event2018_closed_set,
)

# Best hyperparameters from sensitivity analysis
BEST_PARAMS = dict(rho=0.7, r=0.1, sigma=0.1, xi=0.7, phi=0.2, lambda_jsd=1.0)

results = run_tganood_event2012_open_set(
    test_with_one_block=False,   # False = all 21 blocks
    n_epochs=50,
    **BEST_PARAMS
)
```

---

## Hyperparameters

| Parameter | Symbol | Best Value | Description |
|-----------|--------|-----------|-------------|
| Time balance factor | ρ | 0.7 | Blend semantic vs. temporal edge weight |
| Decay rate | r | 0.1 | Controls speed of temporal decay |
| Scaling factor | σ | 0.1 | Sharpness of exponential decay |
| Graph density | ξ | 0.7 | Fraction of top edges to retain |
| GAN ratio | φ | 0.2 | Proportion of GAN samples added to training |
| JSD weight | λ_JSD | 1.0 | Weight of JSD regularisation term |

*Best values determined by sensitivity analysis on Event2012.*

---

## Package Structure

```
tganood/
├── __init__.py          # Package entry point
├── preprocessor.py      # Data loading, splitting, SBERT embedding
├── utils.py             # Text preprocessing, evaluation, JSD utility
├── graph_builder.py     # Temporal graph construction
├── model.py             # GNNEncoder, Generator, Discriminator, ClusterHead
├── losses.py            # GANLoss, JSDLoss, ClusteringLoss
├── train.py             # Joint training loop
└── run.py               # Experiment runner (Event2012/2018, open/closed-set)
```

---

## Results

| Dataset | Setting | AMI | ARI |
|---------|---------|-----|-----|
| Event2012 | Open-set | 0.86 | 0.80 |
| Event2018 | Open-set | 0.84 | 0.79 |

*TGAN-OOD significantly outperforms baselines including HISEvent, HyperSED, FinEvent, QSGNN, KPGNN, EventX, and PP-GCN.*

---

## Citation

If you use this code in your research, please cite the corresponding paper (citation to be added upon publication).

---

## License

This project is released for academic research purposes.
