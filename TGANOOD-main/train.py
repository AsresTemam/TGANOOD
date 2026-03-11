"""
tganood/train.py
~~~~~~~~~~~~~~~~
Joint training loop for TGAN-OOD.

Each training epoch performs:
  1. Build the temporal graph from block embeddings + timestamps.
  2. GNN encoder → node embeddings.
  3. K-means on GNN embeddings → pseudo labels.
  4. ClusterHead forward → clustering loss.
  5. Generator sampling → synthetic embeddings (φ × |train|).
  6. Augmented training set = real + GAN samples.
  7. Discriminator training (k_disc steps).
  8. Generator + JSD adversarial training.
  9. Total loss backprop → update GNN + Generator.
  10. Evaluation: AMI, ARI, JSD.

Optimal hyperparameters:
    rho=0.7, r=0.1, sigma=0.1, xi=0.7, phi=0.2, lambda_jsd=1.0
"""

import torch
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans

from tganood.model import GNNEncoder, Generator, Discriminator, ClusterHead, edges_to_edge_index
from tganood.losses import GANLoss, JSDLoss, ClusteringLoss, compute_total_loss
from tganood.graph_builder import build_temporal_graph
from tganood.utils import evaluate, decode, compute_jsd


# ============================================================
# Helper: k-means pseudo-labelling
# ============================================================

def get_pseudo_labels(embeddings_np, n_clusters, random_state=42):
    """
    Run k-means on GNN embeddings to produce pseudo cluster labels.

    Args:
        embeddings_np : (N, D) numpy array
        n_clusters    : number of clusters
        random_state  : for reproducibility

    Returns:
        (N,) long tensor of pseudo labels
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = km.fit_predict(embeddings_np)
    return torch.tensor(labels, dtype=torch.long)


# ============================================================
# Helper: graph edges → torch_geometric format
# ============================================================

def prepare_graph_tensors(weighted_edges, n_nodes, node_features, device):
    """
    Convert a weighted edge list and raw SBERT embeddings into tensors
    ready for the GNN.

    Args:
        weighted_edges : list of (src, dst, weight) — 1-indexed
        n_nodes        : total number of nodes
        node_features  : (N, D) tensor or numpy array
        device         : torch device

    Returns:
        x           : (N, D) float tensor
        edge_index  : (2, E) long tensor
        edge_weight : (E,) float tensor
    """
    if isinstance(node_features, np.ndarray):
        x = torch.tensor(node_features, dtype=torch.float, device=device)
    else:
        x = node_features.float().to(device)

    edge_index, edge_weight = edges_to_edge_index(weighted_edges, n_nodes, device=device)
    return x, edge_index, edge_weight


# ============================================================
# Main training function
# ============================================================

def train_tganood(
    train_embeddings,
    train_timestamps,
    train_attributes,
    test_embeddings,
    labels_true,
    n_clusters,
    # Temporal graph hyperparameters
    rho=0.7,
    r=0.1,
    sigma=0.1,
    xi=0.7,
    k_neighbours=None,           # if None, use sqrt(N) heuristic
    # GAN hyperparameters
    phi=0.2,                     # GAN sample ratio relative to training size
    latent_dim=128,
    # Loss hyperparameters
    lambda_jsd=1.0,
    # Optimiser settings
    lr_gnn=1e-3,
    lr_gen=1e-3,
    lr_disc=1e-4,
    n_epochs=50,
    k_disc_steps=5,              # discriminator updates per generator update
    # Architecture
    gat_hidden=128,
    gnn_out_dim=128,
    n_heads=4,
    dropout=0.3,
    device='cpu',
    verbose=True,
):
    """
    Full TGAN-OOD training loop.

    Args:
        train_embeddings  : (N, 384) numpy array or tensor — Block 1 SBERT embeddings
        train_timestamps  : (N,) array — numeric timestamps for Block 1 messages
        train_attributes  : list of lists — attribute tokens per message
        test_embeddings   : (M, 384) numpy array or tensor — Block 2 SBERT embeddings
        labels_true       : (N,) list/array — ground-truth event labels (for evaluation)
        n_clusters        : number of expected event clusters
        rho, r, sigma, xi : temporal graph hyperparameters
        k_neighbours      : number of semantic kNN neighbours (None = auto)
        phi               : GAN sample proportion (0.2 = 20% of training size)
        latent_dim        : Generator input noise dimension
        lambda_jsd        : JSD regularisation weight
        lr_gnn, lr_gen, lr_disc : learning rates
        n_epochs          : number of training epochs
        k_disc_steps      : Discriminator updates per Generator step
        gat_hidden, gnn_out_dim, n_heads, dropout : GNN architecture params
        device            : 'cpu' or 'cuda'
        verbose           : print progress

    Returns:
        dict with keys: 'nmi', 'ami', 'ari', 'jsd_pre', 'jsd_post',
                        'prediction', 'history'
    """
    device = torch.device(device)

    # --- Convert embeddings to numpy ---
    if hasattr(train_embeddings, 'numpy'):
        train_emb_np = train_embeddings.numpy()
    else:
        train_emb_np = np.array(train_embeddings)

    if hasattr(test_embeddings, 'numpy'):
        test_emb_np = test_embeddings.numpy()
    else:
        test_emb_np = np.array(test_embeddings)

    N = train_emb_np.shape[0]

    # --- Auto k-neighbours ---
    if k_neighbours is None:
        k_neighbours = max(1, int(np.sqrt(N)))

    # -------------------------------------------------------
    # PRE-GAN: measure distribution shift
    # -------------------------------------------------------
    jsd_pre = compute_jsd(train_emb_np, test_emb_np)
    if verbose:
        print(f'\n[Pre-GAN] Mean JSD = {jsd_pre["mean_jsd"]:.4f}, '
              f'Max JSD = {jsd_pre["max_jsd"]:.4f}')

    # -------------------------------------------------------
    # Build temporal graph
    # -------------------------------------------------------
    if verbose:
        print('[Graph] Building temporal graph...')
    weighted_edges = build_temporal_graph(
        embeddings=train_emb_np,
        timestamps=train_timestamps,
        attributes=train_attributes,
        k=k_neighbours,
        rho=rho, r=r, sigma=sigma, xi=xi,
    )
    if verbose:
        print(f'[Graph] {len(weighted_edges)} edges constructed.')

    # -------------------------------------------------------
    # Instantiate models
    # -------------------------------------------------------
    gnn = GNNEncoder(
        in_dim=train_emb_np.shape[1],
        gat_hidden=gat_hidden,
        out_dim=gnn_out_dim,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)

    generator = Generator(latent_dim=latent_dim, out_dim=train_emb_np.shape[1]).to(device)
    discriminator = Discriminator(in_dim=train_emb_np.shape[1], dropout=dropout).to(device)
    cluster_head = ClusterHead(in_dim=gnn_out_dim, n_clusters=n_clusters).to(device)

    # -------------------------------------------------------
    # Instantiate losses
    # -------------------------------------------------------
    gan_loss_fn = GANLoss()
    jsd_loss_fn = JSDLoss(n_bins=50, bandwidth=0.1)
    cluster_loss_fn = ClusteringLoss()

    # -------------------------------------------------------
    # Optimisers
    # -------------------------------------------------------
    opt_gnn = optim.Adam(
        list(gnn.parameters()) + list(cluster_head.parameters()), lr=lr_gnn
    )
    opt_gen = optim.Adam(generator.parameters(), lr=lr_gen)
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr_disc)

    # -------------------------------------------------------
    # Prepare graph tensors (static per block)
    # -------------------------------------------------------
    x, edge_index, edge_weight = prepare_graph_tensors(
        weighted_edges, N, train_emb_np, device
    )
    test_x = torch.tensor(test_emb_np, dtype=torch.float, device=device)

    # -------------------------------------------------------
    # Training loop
    # -------------------------------------------------------
    history = []
    n_gan_samples = max(1, int(N * phi))

    for epoch in range(1, n_epochs + 1):

        gnn.train()
        generator.train()
        discriminator.train()
        cluster_head.train()

        # === Step A: GNN encode → pseudo labels ===
        with torch.no_grad():
            z_real = gnn(x, edge_index, edge_weight)          # (N, gnn_out_dim)
        pseudo_labels = get_pseudo_labels(
            z_real.cpu().numpy(), n_clusters=n_clusters
        ).to(device)

        # === Step B: Train Discriminator (k_disc_steps) ===
        for _ in range(k_disc_steps):
            opt_disc.zero_grad()
            # Real samples (raw SBERT, not GNN embeddings)
            real_scores = discriminator(x)
            # Fake samples from Generator
            z_noise = torch.randn(n_gan_samples, latent_dim, device=device)
            fake_emb = generator(z_noise).detach()
            fake_scores = discriminator(fake_emb)
            d_loss = gan_loss_fn.discriminator_loss(real_scores, fake_scores)
            d_loss.backward()
            opt_disc.step()

        # === Step C: Augmented training set ===
        z_noise = torch.randn(n_gan_samples, latent_dim, device=device)
        fake_emb = generator(z_noise)                          # (n_gan, emb_dim)
        augmented_x = torch.cat([x, fake_emb], dim=0)         # (N+n_gan, emb_dim)

        # === Step D: Generator adversarial loss ===
        opt_gen.zero_grad()
        fake_scores_for_g = discriminator(fake_emb)
        g_loss = gan_loss_fn.generator_loss(fake_scores_for_g)

        # === Step E: JSD regularisation loss ===
        jsd_reg = jsd_loss_fn(augmented_x[:N], test_x)        # train vs. test

        # === Step F: Clustering loss ===
        opt_gnn.zero_grad()
        z_enc = gnn(x, edge_index, edge_weight)
        logits = cluster_head(z_enc)
        c_loss = cluster_loss_fn(logits, pseudo_labels)

        # === Step G: Combined backward ===
        total_loss, loss_dict = compute_total_loss(c_loss, g_loss, jsd_reg, lambda_jsd)
        total_loss.backward()
        opt_gnn.step()
        opt_gen.step()

        history.append(loss_dict)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f'Epoch {epoch:3d}/{n_epochs} | '
                f'L_total={loss_dict["total"]:.4f} | '
                f'L_cluster={loss_dict["clustering"]:.4f} | '
                f'L_adv={loss_dict["adversarial"]:.4f} | '
                f'L_jsd={loss_dict["jsd"]:.4f}'
            )

    # -------------------------------------------------------
    # Final inference & evaluation
    # -------------------------------------------------------
    gnn.eval()
    with torch.no_grad():
        final_embeddings = gnn(x, edge_index, edge_weight).cpu().numpy()

    # Cluster final embeddings with k-means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    final_labels = km.fit_predict(final_embeddings)

    # Convert to prediction list aligned with node ordering
    # (nodes are 1-indexed internally; final_labels are 0-indexed)
    prediction = list(final_labels)

    nmi, ami, ari = evaluate(labels_true, prediction)

    # POST-GAN JSD
    # Generate synthetic samples at optimal phi and measure JSD
    generator.eval()
    with torch.no_grad():
        gan_emb_final = generator.sample(n_gan_samples, device=str(device)).cpu().numpy()
    aug_for_jsd = np.vstack([train_emb_np, gan_emb_final])
    jsd_post = compute_jsd(aug_for_jsd, test_emb_np)

    if verbose:
        print(f'\n[Results] NMI={nmi:.4f}  AMI={ami:.4f}  ARI={ari:.4f}')
        print(f'[Post-GAN] Mean JSD = {jsd_post["mean_jsd"]:.4f}  '
              f'(reduction: '
              f'{(jsd_pre["mean_jsd"]-jsd_post["mean_jsd"])/jsd_pre["mean_jsd"]*100:.1f}%)')

    return {
        'nmi': nmi,
        'ami': ami,
        'ari': ari,
        'jsd_pre': jsd_pre,
        'jsd_post': jsd_post,
        'prediction': prediction,
        'history': history,
    }
