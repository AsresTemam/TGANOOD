"""
tganood/utils.py
~~~~~~~~~~~~~~~~
Utility functions shared across TGAN-OOD:
  - Text preprocessing (English & French)
  - SBERT embedding
  - Evaluation metrics (NMI, AMI, ARI)
  - Cluster decoding
  - JSD distribution computation
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
)
from scipy.spatial.distance import jensenshannon


# ============================================================
# Text preprocessing
# ============================================================

def replaceAtUser(text):
    """Removes @user mentions and RT prefixes."""
    text = re.sub(r'@[^\s]+|RT @[^\s]+', '', text)
    return text


def removeUnicode(text):
    """Removes unicode escape sequences and non-ASCII characters."""
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    return text


def replaceURL(text):
    """Replaces URLs with 'url' and strips '#' from hashtags."""
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'url', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text


def replaceMultiExclamationMark(text):
    """Collapses repeated exclamation marks."""
    text = re.sub(r'(\!)\1+', '!', text)
    return text


def replaceMultiQuestionMark(text):
    """Collapses repeated question marks."""
    text = re.sub(r'(\?)\1+', '?', text)
    return text


def removeEmoticons(text):
    """Removes common ASCII emoticons."""
    emoticons = r':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-\*|\^\.|\^\-\^|\^\_\^|,\-\)|\)-:|:\'\\(|:\(|:-\(|:\S|T\.T|\._\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O_O|:-@|=/|:/|X\-\(|>.<|>=\(|D:'
    text = re.sub(emoticons, '', text)
    return text


def removeNewLines(text):
    return re.sub(r'\n', '', text)


def preprocess_sentence(s):
    """Full English tweet preprocessing pipeline."""
    return removeNewLines(
        replaceAtUser(
            removeEmoticons(
                replaceMultiQuestionMark(
                    replaceMultiExclamationMark(
                        removeUnicode(replaceURL(s))
                    )
                )
            )
        )
    )


def preprocess_french_sentence(s):
    """French tweet preprocessing (no unicode removal)."""
    return removeNewLines(
        replaceAtUser(
            removeEmoticons(
                replaceMultiQuestionMark(
                    replaceMultiExclamationMark(replaceURL(s))
                )
            )
        )
    )


# ============================================================
# SBERT Embedding
# ============================================================

def SBERT_embed(s_list, language='English'):
    """
    Embed a list of sentences using Sentence-BERT.

    Args:
        s_list   : list of preprocessed strings
        language : 'English' or 'French'

    Returns:
        CPU tensor of shape (len(s_list), 384)
    """
    if language == 'English':
        model = SentenceTransformer('all-MiniLM-L6-v2')
    elif language == 'French':
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    else:
        raise ValueError(f"Unsupported language: {language}")
    embeddings = model.encode(s_list, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings.cpu()


# ============================================================
# Evaluation
# ============================================================

def evaluate(labels_true, labels_pred):
    """
    Compute NMI, AMI, ARI between ground-truth and predicted event labels.

    Returns:
        (nmi, ami, ari) as floats
    """
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    return nmi, ami, ari


def decode(division):
    """
    Convert a cluster division (dict or list of lists) into a flat
    prediction array aligned with original node ordering.

    Args:
        division : dict {cluster_id: [node_ids]} or list [[node_ids], ...]

    Returns:
        list of predicted cluster labels, sorted by node id
    """
    if isinstance(division, dict):
        prediction_dict = {m: event for event, messages in division.items() for m in messages}
    elif isinstance(division, list):
        prediction_dict = {m: event for event, messages in enumerate(division) for m in messages}
    else:
        raise TypeError("division must be a dict or list")
    prediction_dict_sorted = dict(sorted(prediction_dict.items()))
    return list(prediction_dict_sorted.values())


# ============================================================
# JSD Distribution Utility
# ============================================================

def compute_jsd(embeddings_a, embeddings_b, bins=50):
    """
    Compute Jensen-Shannon Divergence between two embedding matrices,
    per embedding dimension, then report aggregate statistics.

    Args:
        embeddings_a : np.ndarray of shape (N, D)
        embeddings_b : np.ndarray of shape (M, D)
        bins         : number of histogram bins per dimension

    Returns:
        dict with keys:
            'mean_jsd'     : float — average JSD across all dimensions
            'max_jsd'      : float — maximum JSD across all dimensions
            'per_dim_jsd'  : list of D floats
    """
    if hasattr(embeddings_a, 'numpy'):
        embeddings_a = embeddings_a.numpy()
    if hasattr(embeddings_b, 'numpy'):
        embeddings_b = embeddings_b.numpy()

    n_dims = embeddings_a.shape[1]
    jsd_values = []
    for dim in range(n_dims):
        hist_a, _ = np.histogram(embeddings_a[:, dim], bins=bins, density=True)
        hist_b, _ = np.histogram(embeddings_b[:, dim], bins=bins, density=True)
        hist_a = hist_a + 1e-10       # avoid log(0)
        hist_b = hist_b + 1e-10
        jsd = jensenshannon(hist_a, hist_b, base=2)
        jsd_values.append(float(jsd))

    return {
        'mean_jsd': float(np.mean(jsd_values)),
        'max_jsd': float(np.max(jsd_values)),
        'per_dim_jsd': jsd_values,
    }