"""
tganood/model.py
~~~~~~~~~~~~~~~~
Neural network modules for TGAN-OOD:

  GNNEncoder   — Graph Attention Network that encodes message nodes into
                 cluster-ready embeddings (replaces SE minimisation).
  Generator    — Produces synthetic SBERT-like embeddings from noise.
  Discriminator— Classifies embeddings as real or GAN-generated.

Dependencies: torch, torch_geometric
  Install: pip install torch torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import torch_geometric; provide a clear error if missing.
try:
    from torch_geometric.nn import GATConv, SAGEConv
    from torch_geometric.utils import from_scipy_sparse_matrix
    _PYGEOM_AVAILABLE = True
except ImportError:
    _PYGEOM_AVAILABLE = False


# ============================================================
# GNN Encoder (Step 4)
# ============================================================

class GNNEncoder(nn.Module):
    """
    2-layer Graph Attention Network (GAT) encoder.

    Maps raw SBERT node features (384-D) to compact cluster embeddings
    (hidden_dim-D, L2-normalised) suitable for downstream clustering.

    Architecture
    ------------
    Layer 1: GATConv(in_dim, gat_hidden, heads=n_heads)  → ReLU → Dropout
    Layer 2: GATConv(gat_hidden*n_heads, out_dim, heads=1) → L2-normalise

    Args:
        in_dim     : input feature dimension (default 384 for SBERT)
        gat_hidden : hidden units per attention head in layer 1
        out_dim    : output embedding dimension
        n_heads    : number of attention heads in layer 1
        dropout    : dropout probability
    """

    def __init__(self, in_dim=384, gat_hidden=128, out_dim=128, n_heads=4, dropout=0.3):
        super().__init__()
        if not _PYGEOM_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNNEncoder. "
                "Install with: pip install torch_geometric"
            )
        self.conv1 = GATConv(in_dim, gat_hidden, heads=n_heads, dropout=dropout)
        self.conv2 = GATConv(gat_hidden * n_heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x           : (N, in_dim) node feature tensor
            edge_index  : (2, E) long tensor of edge indices
            edge_weight : (E,) optional edge weight tensor

        Returns:
            (N, out_dim) L2-normalised node embeddings
        """
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


# ============================================================
# Generator (Step 5)
# ============================================================

class Generator(nn.Module):
    """
    MLP Generator that maps latent noise → synthetic SBERT embeddings.

    Architecture
    ------------
    Linear(latent_dim, 256) → BatchNorm1d → LeakyReLU
    Linear(256, 512)        → BatchNorm1d → LeakyReLU
    Linear(512, out_dim)    → Tanh

    Output is L2-normalised to stay consistent with SBERT embeddings.

    Args:
        latent_dim : dimension of the input noise vector
        out_dim    : output dimension (should match SBERT dimension = 384)
    """

    def __init__(self, latent_dim=128, out_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, out_dim),
            nn.Tanh(),
        )
        self.latent_dim = latent_dim

    def forward(self, z):
        """
        Args:
            z : (B, latent_dim) noise tensor

        Returns:
            (B, out_dim) synthetic embeddings, L2-normalised
        """
        out = self.net(z)
        return F.normalize(out, p=2, dim=-1)

    def sample(self, n, device='cpu'):
        """
        Convenience method: sample n synthetic embeddings.

        Args:
            n      : number of samples to generate
            device : torch device string

        Returns:
            (n, out_dim) detached tensor
        """
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.forward(z)


# ============================================================
# Discriminator (Step 5)
# ============================================================

class Discriminator(nn.Module):
    """
    MLP Discriminator that scores embeddings as real (1) or fake (0).

    Architecture
    ------------
    Linear(in_dim, 256) → LeakyReLU → Dropout
    Linear(256, 128)    → LeakyReLU → Dropout
    Linear(128, 1)      → Sigmoid

    Args:
        in_dim  : input embedding dimension (should match SBERT = 384)
        dropout : dropout probability
    """

    def __init__(self, in_dim=384, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x : (B, in_dim) embedding tensor

        Returns:
            (B, 1) probability scores
        """
        return self.net(x)


# ============================================================
# Cluster head (used in training loop for soft assignments)
# ============================================================

class ClusterHead(nn.Module):
    """
    Lightweight MLP classification head on top of GNN embeddings,
    used to produce soft cluster assignments during training.

    Args:
        in_dim      : GNN output dimension
        n_clusters  : expected number of event clusters
        dropout     : dropout probability
    """

    def __init__(self, in_dim=128, n_clusters=50, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_clusters),
        )

    def forward(self, z):
        """
        Returns:
            (N, n_clusters) logits
        """
        return self.head(z)


# ============================================================
# Helper: convert edge list to torch_geometric edge_index
# ============================================================

def edges_to_edge_index(weighted_edges, n_nodes, device='cpu'):
    """
    Convert a list of (src, dst, weight) tuples to tensors suitable
    for torch_geometric GNN layers.

    Args:
        weighted_edges : list of (src, dst, weight) — 1-indexed nodes
        n_nodes        : total number of nodes (used for validation)
        device         : torch device string

    Returns:
        edge_index  : LongTensor of shape (2, E)
        edge_weight : FloatTensor of shape (E,)
    """
    if not weighted_edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.zeros((0,), dtype=torch.float, device=device)
        return edge_index, edge_weight

    srcs, dsts, weights = zip(*weighted_edges)
    # Convert to 0-indexed
    srcs = torch.tensor([s - 1 for s in srcs], dtype=torch.long, device=device)
    dsts = torch.tensor([d - 1 for d in dsts], dtype=torch.long, device=device)
    # Make undirected
    edge_index = torch.stack(
        [torch.cat([srcs, dsts]), torch.cat([dsts, srcs])], dim=0
    )
    edge_weight = torch.tensor(weights, dtype=torch.float, device=device)
    edge_weight = torch.cat([edge_weight, edge_weight])  # mirror

    return edge_index, edge_weight
