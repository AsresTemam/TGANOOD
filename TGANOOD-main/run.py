"""
tganood/run.py
~~~~~~~~~~~~~~
Experiment runner for TGAN-OOD.

Supported experiment settings:
  - Event2012: open-set (21 daily blocks), closed-set
  - Event2018: open-set (16 daily blocks), closed-set

Usage:
    python -m tganood.run

The runner loads SBERT embeddings, extracts timestamps from the dataset,
and calls train_tganood() for each block, reporting NMI/AMI/ARI and
the pre-/post-GAN JSD reduction.

Optimal hyperparameters:
    rho=0.7, r=0.1, sigma=0.1, xi=0.7, phi=0.2, lambda_jsd=1.0
"""

import pickle
import math
import numpy as np
import pandas as pd
import os
from os.path import exists
from datetime import datetime

from tganood.train import train_tganood
from tganood.graph_builder import search_stable_points


# ============================================================
# Shared helpers
# ============================================================

def load_stable_k(folder, embeddings):
    """
    Return the stable k (number of kNN neighbours).
    Caches result to disk so it is computed only once per block.
    """
    cache_path = os.path.join(folder, 'stable_point.pkl')
    if not exists(cache_path):
        first_k, global_k = search_stable_points(embeddings)
        with open(cache_path, 'wb') as f:
            pickle.dump({'first': first_k, 'global': global_k}, f)
        print('Stable points stored.')
    with open(cache_path, 'rb') as f:
        pts = pickle.load(f)
    print(f'Stable points loaded: first={pts["first"]}, global={pts["global"]}')
    return pts


def timestamps_from_df(df, col='created_at'):
    """Convert a datetime column to UNIX-like float timestamps."""
    try:
        ts = pd.to_datetime(df[col])
        return (ts.astype(np.int64) // 10**9).values.astype(float)
    except Exception:
        # Fallback: sequential integers if parsing fails
        return np.arange(len(df), dtype=float)


def build_node_features_2012(df):
    return [
        [str(u)] + [str(m) for m in um] + [h.lower() for h in hs] + e
        for u, um, hs, e in zip(
            df['user_id'], df['user_mentions'], df['hashtags'], df['entities']
        )
    ]


def build_node_features_2018(df):
    return [
        list(set([str(u)] + [str(m) for m in um] + [h.lower() for h in hs] + e))
        for u, um, hs, e in zip(
            df['user_name'], df['user_mentions'], df['hashtags'], df['entities']
        )
    ]


# ============================================================
# Event2012 — Open-set
# ============================================================

def run_tganood_event2012_open_set(
    test_with_one_block=True,
    rho=0.7, r=0.1, sigma=0.1, xi=0.7,
    phi=0.2, lambda_jsd=1.0,
    n_epochs=50, device='cpu',
    e_a=True, e_s=True,
):
    save_path = './data/Event2012/open_set/'
    if test_with_one_block:
        blocks = [20]
    else:
        blocks = list(range(1, 22))

    all_results = {}
    for block in blocks:
        print(f'\n{"="*54}')
        print(f'Event2012 open-set | Block: {block}  [{datetime.now():%H:%M:%S}]')
        folder = f'{save_path}{block}/'

        # Load embeddings
        with open(os.path.join(folder, 'SBERT_embeddings.pkl'), 'rb') as f:
            embeddings = pickle.load(f)
        if hasattr(embeddings, 'numpy'):
            emb_np = embeddings.numpy()
        else:
            emb_np = np.array(embeddings)

        # Load dataframe
        df_np = np.load(os.path.join(folder, f'{block}.npy'), allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=[
            "original_index", "event_id", "tweet_id", "text", "user_id", "created_at",
            "user_loc", "place_type", "place_full_name", "place_country_code",
            "hashtags", "user_mentions", "image_urls", "entities",
            "words", "filtered_words", "sampled_words", "date"
        ])

        attributes = build_node_features_2012(df)
        timestamps = timestamps_from_df(df, col='created_at')
        labels_true = df['event_id'].tolist()
        n_clusters = len(set(labels_true))
        print(f'n_clusters (GT): {n_clusters}')

        # Stable k selection
        pts = load_stable_k(folder, emb_np)
        k = pts['first'] if e_a else pts['global']
        if k == 0:
            k = max(1, math.ceil((len(emb_np) / 1000) * 10))

        # Load next block for JSD (if available)
        test_block = block + 1
        test_folder = f'{save_path}{test_block}/'
        test_emb_path = os.path.join(test_folder, 'SBERT_embeddings.pkl')
        if exists(test_emb_path):
            with open(test_emb_path, 'rb') as f:
                test_embeddings = pickle.load(f)
        else:
            # Fallback: use the same block (JSD will be 0)
            test_embeddings = embeddings

        results = train_tganood(
            train_embeddings=emb_np,
            train_timestamps=timestamps,
            train_attributes=attributes,
            test_embeddings=test_embeddings,
            labels_true=labels_true,
            n_clusters=n_clusters,
            rho=rho, r=r, sigma=sigma, xi=xi,
            phi=phi, lambda_jsd=lambda_jsd,
            n_epochs=n_epochs, device=device,
        )
        all_results[block] = results

        print(f'Block {block} — NMI={results["nmi"]:.4f}  '
              f'AMI={results["ami"]:.4f}  ARI={results["ari"]:.4f}')
        print(f'  JSD: {results["jsd_pre"]["mean_jsd"]:.4f} → '
              f'{results["jsd_post"]["mean_jsd"]:.4f}')

    return all_results


# ============================================================
# Event2012 — Closed-set
# ============================================================

def run_tganood_event2012_closed_set(
    rho=0.7, r=0.1, sigma=0.1, xi=0.7,
    phi=0.2, lambda_jsd=1.0,
    n_epochs=50, device='cpu',
    e_a=True, e_s=True,
):
    save_path = './data/Event2012/closed_set/'
    print(f'\n{"="*54}')
    print(f'Event2012 closed-set  [{datetime.now():%H:%M:%S}]')

    # Load test set
    test_df_np = np.load(save_path + 'test_set.npy', allow_pickle=True)
    df = pd.DataFrame(data=test_df_np, columns=[
        "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
        "place_type", "place_full_name", "place_country_code", "hashtags",
        "user_mentions", "image_urls", "entities", "words", "filtered_words", "sampled_words"
    ])
    print('Dataframe loaded.')

    with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    emb_np = embeddings.numpy() if hasattr(embeddings, 'numpy') else np.array(embeddings)

    attributes = build_node_features_2012(df)
    timestamps = timestamps_from_df(df, col='created_at')
    labels_true = df['event_id'].tolist()
    n_clusters = len(set(labels_true))
    print(f'n_clusters (GT): {n_clusters}')

    pts = load_stable_k(save_path, emb_np)
    k = pts['first']
    if k == 0:
        k = max(1, math.ceil((len(emb_np) / 1000) * 10))

    # In closed-set, use the same embeddings as test reference
    results = train_tganood(
        train_embeddings=emb_np,
        train_timestamps=timestamps,
        train_attributes=attributes,
        test_embeddings=emb_np,
        labels_true=labels_true,
        n_clusters=n_clusters,
        rho=rho, r=r, sigma=sigma, xi=xi,
        phi=phi, lambda_jsd=lambda_jsd,
        n_epochs=n_epochs, device=device,
    )
    print(f'NMI={results["nmi"]:.4f}  AMI={results["ami"]:.4f}  ARI={results["ari"]:.4f}')
    return results


# ============================================================
# Event2018 — Open-set
# ============================================================

def run_tganood_event2018_open_set(
    test_with_one_block=True,
    rho=0.7, r=0.1, sigma=0.1, xi=0.7,
    phi=0.2, lambda_jsd=1.0,
    n_epochs=50, device='cpu',
    e_a=True, e_s=True,
):
    save_path = './data/Event2018/open_set/'
    if test_with_one_block:
        blocks = [16]
    else:
        blocks = list(range(1, 17))

    all_results = {}
    for block in blocks:
        print(f'\n{"="*54}')
        print(f'Event2018 open-set | Block: {block}  [{datetime.now():%H:%M:%S}]')
        folder = f'{save_path}{block}/'

        with open(os.path.join(folder, 'SBERT_embeddings.pkl'), 'rb') as f:
            embeddings = pickle.load(f)
        emb_np = embeddings.numpy() if hasattr(embeddings, 'numpy') else np.array(embeddings)

        df_np = np.load(os.path.join(folder, f'{block}.npy'), allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=[
            "original_index", "tweet_id", "user_name", "text", "time", "event_id",
            "user_mentions", "hashtags", "urls", "words", "created_at",
            "filtered_words", "entities", "sampled_words", "date"
        ])

        attributes = build_node_features_2018(df)
        timestamps = timestamps_from_df(df, col='created_at')
        labels_true = df['event_id'].tolist()
        n_clusters = len(set(labels_true))
        print(f'n_clusters (GT): {n_clusters}')

        pts = load_stable_k(folder, emb_np)
        k = pts['first'] if e_a else pts['global']
        if k == 0:
            k = max(1, math.ceil((len(emb_np) / 1000) * 10))

        test_block = block + 1
        test_folder = f'{save_path}{test_block}/'
        test_emb_path = os.path.join(test_folder, 'SBERT_embeddings.pkl')
        if exists(test_emb_path):
            with open(test_emb_path, 'rb') as f:
                test_embeddings = pickle.load(f)
        else:
            test_embeddings = embeddings

        results = train_tganood(
            train_embeddings=emb_np,
            train_timestamps=timestamps,
            train_attributes=attributes,
            test_embeddings=test_embeddings,
            labels_true=labels_true,
            n_clusters=n_clusters,
            rho=rho, r=r, sigma=sigma, xi=xi,
            phi=phi, lambda_jsd=lambda_jsd,
            n_epochs=n_epochs, device=device,
        )
        all_results[block] = results
        print(f'Block {block} — NMI={results["nmi"]:.4f}  '
              f'AMI={results["ami"]:.4f}  ARI={results["ari"]:.4f}')
        print(f'  JSD: {results["jsd_pre"]["mean_jsd"]:.4f} → '
              f'{results["jsd_post"]["mean_jsd"]:.4f}')

    return all_results


# ============================================================
# Event2018 — Closed-set
# ============================================================

def run_tganood_event2018_closed_set(
    rho=0.7, r=0.1, sigma=0.1, xi=0.7,
    phi=0.2, lambda_jsd=1.0,
    n_epochs=50, device='cpu',
    e_a=True, e_s=True,
):
    save_path = './data/Event2018/closed_set/'
    print(f'\n{"="*54}')
    print(f'Event2018 closed-set  [{datetime.now():%H:%M:%S}]')

    test_df_np = np.load(save_path + 'test_set.npy', allow_pickle=True)
    df = pd.DataFrame(data=test_df_np, columns=[
        "tweet_id", "user_name", "text", "time", "event_id", "user_mentions",
        "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words"
    ])
    print('Dataframe loaded.')

    with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    emb_np = embeddings.numpy() if hasattr(embeddings, 'numpy') else np.array(embeddings)

    attributes = build_node_features_2018(df)
    timestamps = timestamps_from_df(df, col='created_at')
    labels_true = df['event_id'].tolist()
    n_clusters = len(set(labels_true))
    print(f'n_clusters (GT): {n_clusters}')

    pts = load_stable_k(save_path, emb_np)
    k = pts['first']
    if k == 0:
        k = max(1, math.ceil((len(emb_np) / 1000) * 10))

    results = train_tganood(
        train_embeddings=emb_np,
        train_timestamps=timestamps,
        train_attributes=attributes,
        test_embeddings=emb_np,
        labels_true=labels_true,
        n_clusters=n_clusters,
        rho=rho, r=r, sigma=sigma, xi=xi,
        phi=phi, lambda_jsd=lambda_jsd,
        n_epochs=n_epochs, device=device,
    )
    print(f'NMI={results["nmi"]:.4f}  AMI={results["ami"]:.4f}  ARI={results["ari"]:.4f}')
    return results


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    # Best hyperparameters from sensitivity analysis
    BEST_PARAMS = dict(rho=0.7, r=0.1, sigma=0.1, xi=0.7, phi=0.2, lambda_jsd=1.0)

    # Uncomment the experiment you want to run:

    # --- Event2012 ---
    # run_tganood_event2012_open_set(test_with_one_block=True, n_epochs=50, **BEST_PARAMS)
    run_tganood_event2012_closed_set(n_epochs=50, **BEST_PARAMS)

    # --- Event2018 ---
    # run_tganood_event2018_open_set(test_with_one_block=True, n_epochs=50, **BEST_PARAMS)
    # run_tganood_event2018_closed_set(n_epochs=50, **BEST_PARAMS)