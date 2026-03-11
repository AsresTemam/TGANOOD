"""
tganood/graph_builder.py
~~~~~~~~~~~~~~~~~~~~~~~~
Temporal graph construction for TGAN-OOD.

Builds a dual-edge message graph (attribute + semantic kNN) with temporal weighting:
  - Time balance factor  rho  (ρ): blend temporal vs. semantic weight
  - Decay rate           r       : exponential decay over time difference
  - Scaling factor       sigma (σ): sharpness of the decay
  - Graph density        xi    (ξ): fraction of top edges to retain

Edge weight formula:
    w(i,j) = ρ × w_semantic(i,j) + (1 - ρ) × exp(-r × |Δt|^σ)

where Δt = |t_i - t_j| in seconds (or any consistent unit).
"""

import numpy as np
from itertools import combinations


# ============================================================
# Attribute-based edges
# ============================================================

def get_attribute_edges(attributes):
    """
    Build edges connecting messages that share at least one attribute
    (hashtag, user-mention, entity, etc.).

    Args:
        attributes : list of lists — each inner list contains the
                     attribute tokens for one message.

    Returns:
        list of (src, dst) tuples (1-indexed, undirected, deduplicated)
    """
    attr_nodes_dict = {}
    for i, attr_list in enumerate(attributes):
        for attr in attr_list:
            attr_nodes_dict.setdefault(attr, []).append(i + 1)  # 1-indexed

    for key in attr_nodes_dict:
        attr_nodes_dict[key].sort()

    edges = []
    for nodes in attr_nodes_dict.values():
        edges.extend(combinations(nodes, 2))
    return list(set(edges))


# ============================================================
# Semantic kNN edges
# ============================================================

def get_semantic_knn_edges(embeddings, k):
    """
    Build k-NN edges weighted by Pearson correlation between embeddings.

    Args:
        embeddings : np.ndarray or torch.Tensor of shape (N, D)
        k          : number of nearest neighbours per node

    Returns:
        list of (src, dst, weight) tuples (1-indexed, positive weight)
    """
    if hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()

    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    sorted_indices = np.argsort(corr_matrix)

    edges = set()
    for i in range(k):
        dst_ids = sorted_indices[:, -(i + 1)]
        for s, d in enumerate(dst_ids):
            w = corr_matrix[s, d]
            if w > 0:
                u, v = (s + 1, d + 1) if s < d else (d + 1, s + 1)
                edges.add((u, v, float(w)))
    return list(edges)


# ============================================================
# Temporal weighting
# ============================================================

def compute_temporal_weight(t_i, t_j, r, sigma):
    """
    Compute the temporal proximity weight between two messages.

        w_temporal(i,j) = exp(-r × |t_i - t_j|^sigma)

    Args:
        t_i, t_j : timestamps as numeric values (e.g., UNIX seconds)
        r        : decay rate  — higher means faster decay
        sigma    : scaling factor — controls sharpness of decay

    Returns:
        float in (0, 1]
    """
    delta_t = abs(t_i - t_j)
    return float(np.exp(-r * (delta_t ** sigma)))


def blend_edge_weights(semantic_edges, timestamps, rho, r, sigma):
    """
    Blend semantic and temporal weights for each kNN edge.

        w_final(i,j) = ρ × w_semantic + (1 - ρ) × w_temporal

    Args:
        semantic_edges : list of (src, dst, w_semantic) — 1-indexed
        timestamps     : list or array of per-message timestamps (length N)
        rho            : time balance factor ρ ∈ [0, 1]
        r              : decay rate
        sigma          : scaling factor

    Returns:
        list of (src, dst, w_final) tuples
    """
    blended = []
    for src, dst, w_sem in semantic_edges:
        t_i = timestamps[src - 1]  # convert back to 0-index
        t_j = timestamps[dst - 1]
        w_temp = compute_temporal_weight(t_i, t_j, r, sigma)
        w_final = rho * w_sem + (1.0 - rho) * w_temp
        blended.append((src, dst, float(w_final)))
    return blended


# ============================================================
# Stable-point detection using 1D entropy proxy
# ============================================================

def _calc_entropy(corr_matrix, k, sorted_indices):
    """
    Approximate 1D entropy for the k-NN graph (degree-based proxy).
    Used internally by search_stable_points.
    """
    n = corr_matrix.shape[0]
    degree = np.zeros(n)
    for i in range(k):
        dst_ids = sorted_indices[:, -(i + 1)]
        for s, d in enumerate(dst_ids):
            if corr_matrix[s, d] > 0:
                degree[s] += corr_matrix[s, d]
                degree[d] += corr_matrix[s, d]
    vol = degree.sum()
    if vol == 0:
        return 0.0
    ent = 0.0
    for deg in degree:
        if deg > 0:
            p = deg / vol
            ent -= p * np.log(p + 1e-12)
    return ent


def search_stable_points(embeddings, max_k=200):
    """
    Automatically determine the optimal number of nearest neighbours k
    by finding stable minima in the 1D entropy curve.

    Args:
        embeddings : np.ndarray (N, D)
        max_k      : maximum k to search

    Returns:
        (first_stable_k, global_stable_k) — both are 1-indexed neighbour counts
    """
    if hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()

    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    sorted_indices = np.argsort(corr_matrix)

    entropies = [_calc_entropy(corr_matrix, k + 1, sorted_indices) for k in range(max_k)]

    stable = []
    for i in range(1, len(entropies) - 1):
        if entropies[i] < entropies[i - 1] and entropies[i] < entropies[i + 1]:
            stable.append(i)

    if not stable:
        print(f'No stable points found after checking k=1 to {max_k}.')
        return 0, 0

    stable_vals = [entropies[i] for i in stable]
    global_idx = stable[stable_vals.index(min(stable_vals))]
    print(f'Stable points at k: {[s+1 for s in stable]}')
    print(f'First stable k={stable[0]+1}, global stable k={global_idx+1}')
    return stable[0] + 1, global_idx + 1


# ============================================================
# Main graph construction function
# ============================================================

def build_temporal_graph(
    embeddings,
    timestamps,
    attributes,
    k,
    rho=0.7,
    r=0.1,
    sigma=0.1,
    xi=0.7,
    e_a=True,
    e_s=True,
):
    """
    Build a temporally-weighted message graph for TGAN-OOD.

    Args:
        embeddings : np.ndarray or tensor of shape (N, 384)
        timestamps : list/array of length N — numeric timestamps
                     (e.g., UNIX seconds or integer day indices)
        attributes : list of lists — attribute tokens per message
        k          : number of semantic kNN neighbours
        rho        : time balance factor ρ (0=all temporal, 1=all semantic)
        r          : temporal decay rate
        sigma      : temporal scaling factor
        xi         : graph density — fraction of top-weighted edges to retain (0,1]
        e_a        : whether to include attribute-based edges
        e_s        : whether to include semantic kNN edges

    Returns:
        list of (src, dst, weight) tuples — weighted, undirected graph edges
    """
    if hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()
    timestamps = np.array(timestamps, dtype=float)

    # Normalise timestamps to [0, 1] so r is scale-independent
    t_range = timestamps.max() - timestamps.min()
    if t_range > 0:
        timestamps_norm = (timestamps - timestamps.min()) / t_range
    else:
        timestamps_norm = timestamps * 0.0  # all same time → no decay

    all_edges = []

    # 1. Attribute-based edges (weighted by embedding correlation)
    if e_a:
        attr_edges = get_attribute_edges(attributes)
        corr_matrix = np.corrcoef(embeddings)
        np.fill_diagonal(corr_matrix, 0)
        weighted_attr_edges = [
            (s, d, float(corr_matrix[s - 1, d - 1]))
            for s, d in attr_edges
            if corr_matrix[s - 1, d - 1] > 0
        ]
        all_edges.extend(weighted_attr_edges)

    # 2. Semantic kNN edges with temporal blending
    if e_s:
        sem_edges = get_semantic_knn_edges(embeddings, k)
        if rho < 1.0:
            blended_edges = blend_edge_weights(sem_edges, timestamps_norm, rho, r, sigma)
        else:
            blended_edges = sem_edges
        all_edges.extend(blended_edges)

    # Deduplicate (keep max weight for duplicate pairs)
    edge_dict = {}
    for src, dst, w in all_edges:
        key = (min(src, dst), max(src, dst))
        edge_dict[key] = max(edge_dict.get(key, -1), w)
    all_edges = [(s, d, w) for (s, d), w in edge_dict.items() if w > 0]

    # Apply graph density control (xi): retain top-xi fraction of edges by weight
    if xi is not None and 0 < xi < 1.0:
        all_edges_sorted = sorted(all_edges, key=lambda e: e[2], reverse=True)
        n_keep = max(1, int(len(all_edges_sorted) * xi))
        all_edges = all_edges_sorted[:n_keep]

    return all_edges
