import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple
import torch

def compute_far_frr(
    scores_pos: NDArray[np.float64],
    scores_neg: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    scores_pos_sorted = np.sort(scores_pos)
    scores_neg_sorted = np.sort(scores_neg)

    pos_count_less = np.searchsorted(scores_pos_sorted, thresholds, side='left')
    neg_count_less = np.searchsorted(scores_neg_sorted, thresholds, side='left')

    frr = pos_count_less / len(scores_pos)
    far = (len(scores_neg) - neg_count_less) / len(scores_neg)

    return far, frr


def compute_eer(
    scores_pos: NDArray[np.float64],
    scores_neg: NDArray[np.float64],
) -> Tuple[float, float]:
    scores_pos = np.asarray(scores_pos, dtype=np.float64)
    scores_neg = np.asarray(scores_neg, dtype=np.float64)

    if len(scores_pos) == 0 or len(scores_neg) == 0:
        raise ValueError(
            f"Cannot compute EER: Got {len(scores_pos)} positive and {len(scores_neg)} negative scores."
        )

    all_scores = np.concatenate([scores_pos, scores_neg])
    thresholds = np.unique(all_scores)

    far, frr = compute_far_frr(scores_pos, scores_neg, thresholds)

    diff = far - frr

    if np.all(diff >= 0) or np.all(diff <= 0):
        idx = np.argmin(np.abs(diff))
        eer = (far[idx] + frr[idx]) / 2.0
        return float(eer), float(thresholds[idx])

    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        idx = np.argmin(np.abs(diff))
        eer = (far[idx] + frr[idx]) / 2.0
        return float(eer), float(thresholds[idx])

    idx = sign_changes[0]

    d_far = far[idx + 1] - far[idx]
    d_frr = frr[idx + 1] - frr[idx]
    denominator = d_far - d_frr

    if abs(denominator) < 1e-12:
        eer = (far[idx] + frr[idx]) / 2.0
        threshold = thresholds[idx]
    else:
        alpha = (frr[idx] - far[idx]) / denominator
        alpha = np.clip(alpha, 0.0, 1.0)
        eer = far[idx] + d_far * alpha
        threshold = thresholds[idx] + (thresholds[idx + 1] - thresholds[idx]) * alpha

    return float(eer), float(threshold)


def compute_delta_eer(eer_original: float, eer_privatized: float) -> float:
    return eer_privatized - eer_original


def construct_verification_pairs(
    query_embeddings: NDArray[np.float64],
    candidate_embeddings: NDArray[np.float64],
    query_labels: List[str],
    candidate_labels: List[str],
    ground_truth_pairs: List[int],
    similarity_matrix: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    if similarity_matrix is None:
        q_norm = query_embeddings / (
            np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12
        )
        c_norm = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-12
        )
        similarity_matrix = q_norm @ c_norm.T

    similarity_matrix = similarity_matrix.flatten()
    
    if not isinstance(ground_truth_pairs, np.ndarray):
        ground_truth_pairs = np.asarray(ground_truth_pairs)

    pos_mask = ground_truth_pairs == 1
    neg_mask = ground_truth_pairs == 0

    scores_pos = similarity_matrix[pos_mask]
    scores_neg = similarity_matrix[neg_mask]

    return scores_pos.astype(np.float64), scores_neg.astype(np.float64)


def compute_eer_from_embeddings(
    query_embeddings: NDArray[np.float64],
    candidate_embeddings: NDArray[np.float64],
    query_labels: List[str],
    candidate_labels: List[str],
    ground_truth_pairs: List[int],
) -> Dict[str, float]:

    scores_pos, scores_neg = construct_verification_pairs(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        ground_truth_pairs=ground_truth_pairs,
        query_labels=query_labels,
        candidate_labels=candidate_labels,
    )

    eer, threshold = compute_eer(scores_pos, scores_neg)

    return {
        "eer": eer,
        "threshold": threshold,
        "n_positive_pairs": len(scores_pos),
        "n_negative_pairs": len(scores_neg),
    }



