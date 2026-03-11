"""
tganood — TGAN-OOD: Temporal GAN for Out-of-Distribution Social Event Detection

Package structure:
  preprocessor.py  — Data loading, SBERT embedding, dataset splitting
  utils.py         — Text preprocessing, evaluation metrics, JSD utility
  graph_builder.py — Temporal graph construction (ρ, r, σ, ξ parameters)
  model.py         — GNNEncoder (GAT), Generator, Discriminator, ClusterHead
  losses.py        — GANLoss, JSDLoss, ClusteringLoss, compute_total_loss
  train.py         — Joint training loop (GNN + GAN + JSD regularisation)
  run.py           — Experiment runner for Event2012/2018 open/closed-set
"""

from tganood.utils import evaluate, decode, compute_jsd, SBERT_embed
from tganood.graph_builder import build_temporal_graph, search_stable_points
from tganood.model import GNNEncoder, Generator, Discriminator, ClusterHead
from tganood.losses import GANLoss, JSDLoss, ClusteringLoss, compute_total_loss
from tganood.train import train_tganood

__all__ = [
    # utils
    'evaluate', 'decode', 'compute_jsd', 'SBERT_embed',
    # graph_builder
    'build_temporal_graph', 'search_stable_points',
    # model
    'GNNEncoder', 'Generator', 'Discriminator', 'ClusterHead',
    # losses
    'GANLoss', 'JSDLoss', 'ClusteringLoss', 'compute_total_loss',
    # train
    'train_tganood',
]
