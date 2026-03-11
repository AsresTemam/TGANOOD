"""
tganood/losses.py
~~~~~~~~~~~~~~~~~
Loss functions for TGAN-OOD training:

  1. GAN adversarial loss (BCE — standard binary cross-entropy)
  2. JSD regularisation loss (differentiable soft-histogram JSD)
  3. Combined total loss

The total optimisation objective:

    L_total = L_clustering + L_adv + λ_JSD × L_jsd

where L_clustering is the cross-entropy on cluster assignments,
L_adv is the standard GAN adversarial loss, and L_jsd penalises the
distribution gap between augmented training and future test embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 1. GAN Adversarial Loss
# ============================================================

class GANLoss(nn.Module):
    """
    Standard binary cross-entropy GAN loss.

    Discriminator objective:
        L_D = -E[log D(x_real)] - E[log(1 - D(G(z)))]

    Generator objective:
        L_G = -E[log D(G(z))]   (non-saturating variant)
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def discriminator_loss(self, real_scores, fake_scores):
        """
        Compute the discriminator loss.

        Args:
            real_scores : (B, 1) D output on real embeddings
            fake_scores : (B, 1) D output on G(z) embeddings

        Returns:
            scalar loss tensor
        """
        real_labels = torch.ones_like(real_scores)
        fake_labels = torch.zeros_like(fake_scores)
        loss_real = self.bce(real_scores, real_labels)
        loss_fake = self.bce(fake_scores, fake_labels)
        return (loss_real + loss_fake) * 0.5

    def generator_loss(self, fake_scores):
        """
        Compute the generator (non-saturating) loss.

        Args:
            fake_scores : (B, 1) D output on G(z) embeddings

        Returns:
            scalar loss tensor
        """
        real_labels = torch.ones_like(fake_scores)
        return self.bce(fake_scores, real_labels)


# ============================================================
# 2. Differentiable JSD Regularisation Loss
# ============================================================

class JSDLoss(nn.Module):
    """
    Differentiable Jensen-Shannon Divergence loss computed via
    soft histograms over each embedding dimension.

    Soft histogram approach:
        For each dimension d:
            - Place n_bins Gaussian kernels uniformly over the
              combined value range.
            - The soft count of sample x in bin b is:
                  K(x, μ_b) = exp( -(x - μ_b)² / (2σ²) )
            - Normalise to obtain a probability distribution P (or Q).
            - Compute JSD(P || Q) using the Shannon entropy formula.
        Average JSD across all dimensions.

    This loss is fully differentiable through the embedding values,
    so it can be backpropagated into the GNN and Generator.

    Args:
        n_bins   : number of histogram bins (default 50)
        bandwidth: Gaussian kernel bandwidth (default 0.1)
    """

    def __init__(self, n_bins=50, bandwidth=0.1):
        super().__init__()
        self.n_bins = n_bins
        self.bandwidth = bandwidth

    def _soft_histogram(self, x, bin_centers):
        """
        Compute a soft (differentiable) normalised histogram.

        Args:
            x           : (N,) 1-D tensor of values
            bin_centers : (B,) tensor of bin centre positions

        Returns:
            (B,) probability vector
        """
        # x: (N, 1), bin_centers: (1, B)
        diffs = x.unsqueeze(1) - bin_centers.unsqueeze(0)          # (N, B)
        weights = torch.exp(-0.5 * (diffs / self.bandwidth) ** 2)  # (N, B)
        hist = weights.sum(dim=0)                                    # (B,)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        return hist

    def _jsd_1d(self, p, q):
        """
        Compute JSD between two discrete distributions p and q.

        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M),  M = 0.5*(P+Q)

        Returns scalar tensor.
        """
        m = 0.5 * (p + q) + 1e-10
        kl_pm = (p * (torch.log(p + 1e-10) - torch.log(m))).sum()
        kl_qm = (q * (torch.log(q + 1e-10) - torch.log(m))).sum()
        return 0.5 * (kl_pm + kl_qm)

    def forward(self, emb_train, emb_test):
        """
        Compute mean JSD across all embedding dimensions.

        Args:
            emb_train : (N, D) augmented training embeddings
            emb_test  : (M, D) test / next-block embeddings

        Returns:
            scalar JSD loss (differentiable)
        """
        D = emb_train.shape[1]
        jsd_total = torch.tensor(0.0, device=emb_train.device, requires_grad=False)

        for d in range(D):
            x_train = emb_train[:, d]
            x_test = emb_test[:, d]

            # Define bin centres over the combined range
            x_all = torch.cat([x_train.detach(), x_test.detach()])
            x_min, x_max = x_all.min().item(), x_all.max().item()
            if x_max == x_min:
                continue
            bin_centers = torch.linspace(
                x_min, x_max, self.n_bins, device=emb_train.device
            )

            p = self._soft_histogram(x_train, bin_centers)
            q = self._soft_histogram(x_test.detach(), bin_centers)  # test is reference
            jsd_total = jsd_total + self._jsd_1d(p, q)

        return jsd_total / D


# ============================================================
# 3. Clustering Loss (cross-entropy on cluster assignments)
# ============================================================

class ClusteringLoss(nn.Module):
    """
    Cross-entropy loss for cluster assignment.

    During the training loop, pseudo-labels from k-means on GNN
    embeddings are used as supervision for the ClusterHead.

    Args:
        label_smoothing : smoothing coefficient (default 0.0)
    """

    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, pseudo_labels):
        """
        Args:
            logits        : (N, n_clusters) from ClusterHead
            pseudo_labels : (N,) long tensor of pseudo cluster ids

        Returns:
            scalar cross-entropy loss
        """
        return self.ce(logits, pseudo_labels)


# ============================================================
# 4. Combined Loss
# ============================================================

def compute_total_loss(
    cluster_loss,
    gen_loss,
    jsd_loss_val,
    lambda_jsd=1.0,
):
    """
    Combine all loss terms into the total TGAN-OOD objective.

        L_total = L_clustering + L_adv + λ_JSD × L_jsd

    Args:
        cluster_loss  : scalar — clustering cross-entropy loss
        gen_loss      : scalar — GAN generator adversarial loss
        jsd_loss_val  : scalar — JSD regularisation loss
        lambda_jsd    : float — weight for JSD term (best = 1.0 from sensitivity)

    Returns:
        total_loss    : scalar tensor
        loss_dict     : dict of individual loss values for logging
    """
    total = cluster_loss + gen_loss + lambda_jsd * jsd_loss_val
    loss_dict = {
        'clustering': cluster_loss.item() if hasattr(cluster_loss, 'item') else float(cluster_loss),
        'adversarial': gen_loss.item() if hasattr(gen_loss, 'item') else float(gen_loss),
        'jsd': jsd_loss_val.item() if hasattr(jsd_loss_val, 'item') else float(jsd_loss_val),
        'total': total.item() if hasattr(total, 'item') else float(total),
    }
    return total, loss_dict
