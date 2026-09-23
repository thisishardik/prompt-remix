import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Dict, List

from scipy.spatial.distance import pdist, squareform

def choose_sliders_round_robin(styles_normalized, top_n, available_types, all_types, standard_devs, closest=False, mix_toward=True):
    def distance_metric(a, b):
        diff = a - b
        diff = (diff / standard_devs).fillna(0)
        return np.linalg.norm(diff)
    
    pairwise_dist = squareform(pdist(styles_normalized, distance_metric))
    sliders = {}
    author_idx_used = set()
    author_idx = 0
    while len(sliders) < len(styles_normalized):
        author_idx_used.add(author_idx)
        author = styles_normalized.index[author_idx]
        ranked_authors = np.argsort(pairwise_dist[author_idx])
        if not closest:
            ranked_authors = ranked_authors[::-1]

        next_author_idx = next((idx for idx in ranked_authors if idx not in author_idx_used), 0)
        author_diff = styles_normalized.iloc[next_author_idx] - styles_normalized.iloc[author_idx]
        author_diff = (author_diff / standard_devs).fillna(0)
        candidate_axes = np.abs(author_diff.values).argsort()[::-1][:top_n + 4]
        num_chosen = 0
        type_chosen = False
        author_sliders = {}
        for axis in candidate_axes:
            effective_axis = author_diff.index[axis]
            n_std = np.abs(author_diff[effective_axis])
            weight = np.sign(author_diff[effective_axis])
            if not mix_toward:
                weight *= -1.0

            if effective_axis in all_types:
                types_to_choose = [type_name for type_name in available_types if type_name != effective_axis]
                if len(types_to_choose) == 0 or type_chosen:
                    continue
                effective_axis = np.random.choice(types_to_choose)
                type_chosen = True
                weight = 1.0
            weight *= std_to_weight(n_std)
            author_sliders[effective_axis] = weight
            num_chosen += 1
            if num_chosen >= top_n:
                break
        
        sliders[author] = author_sliders
        author_idx = next_author_idx

    return pd.Series(sliders)

def std_to_weight(n_std: float) -> float:
    if n_std <= 1:
        return 0.5
    elif n_std > 1 and n_std <= 2:
        return 0.7
    elif n_std > 2 and n_std <= 3:
        return 0.9
    else:
        return 1.0
    
def choose_directions_genre_mean(row: pd.Series, top_n: int, available_types: List[str], all_types: List[str]) -> Dict[str, float]:
    indices = np.abs(row.values).argsort()[::-1][:(top_n + 4)]
    num_chosen = 0
    axes = row.index[indices]
    directions = {}
    type_chosen = False
    for axis in axes:
        effective_axis = axis
        n_std = np.abs(row[axis])
        weight = (-1.0) * np.sign(row[axis])

        if axis in all_types:
            types_to_choose = [type_name for type_name in available_types if type_name != axis]
            if len(types_to_choose) == 0 or type_chosen:
                continue
            effective_axis = np.random.choice(types_to_choose)
            type_chosen = True
            weight = 1.0


        weight *= std_to_weight(n_std)
        directions[effective_axis] = weight
        num_chosen += 1
        if num_chosen >= top_n:
            break

    return directions

def choose_directions_target_author(
    styles_normalized: pd.DataFrame,
    top_n: int,
    available_types: List[str],
    all_types: List[str],
    standard_devs: pd.Series,
    target_author_id: str = "target_author",
    mix_toward: bool = True,
) -> pd.Series:
    """
    For each author, choose up to `top_n` slider directions that move their style
    toward (or away from) the `target_author_id`, using the same axis scaling
    conventions as choose_sliders_round_robin.

    Parameters
    ----------
    styles_normalized:
        DataFrame indexed by author_id with columns = style axes (already normalized).
        Must contain a row for `target_author_id`.
    top_n:
        number of axes/directions to emit per author.
    available_types:
        subset of `all_types` that have non-zero stddev and are allowed to be selected.
    all_types:
        list of type axes (e.g. type_persuasive, type_narrative, ...).
    standard_devs:
        per-axis standard deviations used to scale diffs.
    target_author_id:
        index value that corresponds to the target author row.
    mix_toward:
        if True: choose directions that move authors toward the target.
        if False: move away.

    Returns
    -------
    pd.Series:
        index = author ids (excluding target), value = dict(axis -> weight)
    """
    if target_author_id not in styles_normalized.index:
        raise KeyError(
            f"styles_normalized must contain a '{target_author_id}' row to do style transfer."
        )

    target_vec = styles_normalized.loc[target_author_id]

    best_target_type = None
    available_types_in_df = [t for t in available_types if t in styles_normalized.columns]
    if len(available_types_in_df) > 0:
        best_target_type = target_vec[available_types_in_df].sort_values(ascending=False).index[0]

    sliders: Dict[str, Dict[str, float]] = {}

    for author_id, author_vec in styles_normalized.iterrows():
        if author_id == target_author_id:
            continue

        author_diff = target_vec - author_vec
        if not mix_toward:
            author_diff = -author_diff

        author_diff = (author_diff / standard_devs).fillna(0)

        candidate_axes = author_diff.abs().sort_values(ascending=False).index[: (top_n + 4)]

        num_chosen = 0
        type_chosen = False
        author_sliders: Dict[str, float] = {}

        for axis in candidate_axes:
            n_std = float(abs(author_diff[axis]))
            if n_std == 0.0:
                continue

            effective_axis = axis
            weight = float(np.sign(author_diff[axis]))

            if axis in all_types:
                if type_chosen or best_target_type is None:
                    continue
                effective_axis = best_target_type
                weight = 1.0
                type_chosen = True

            weight *= std_to_weight(n_std)
            author_sliders[effective_axis] = weight
            num_chosen += 1

            if num_chosen >= top_n:
                break

        sliders[author_id] = author_sliders

    return pd.Series(sliders)

def combine_directions(directions: List[str], all_types: List[str]) -> Dict[str, float]:
    direction_sums = defaultdict(float)
    direction_counts = defaultdict(int)
    type_chosen = None
    for slider in directions:
        for axis, weight in slider.items():
            if axis in all_types:
                if type_chosen is not None and type_chosen != axis:
                    continue
                type_chosen = axis
            direction_sums[axis] += weight
            direction_counts[axis] += 1
            
    combined_directions = {axis: weight / direction_counts[axis] for axis, weight in direction_sums.items()}
    return combined_directions


def choose_random_directions(
    styles_normalized: pd.DataFrame,
    top_n: int,
    categorical_types: List[str],
    all_axes: List[str],
    mix_toward: bool = True
) -> pd.Series:
    sliders: Dict[str, Dict[str, float]] = {}
    
    non_categorical_axes = [col for col in all_axes if col not in categorical_types]
    
    for author_id in styles_normalized.index:
        author_sliders: Dict[str, float] = {}
        num_chosen = 0
        categorical_chosen = False
        
        shuffled_non_cat = np.random.permutation(non_categorical_axes).tolist()
        shuffled_cat = np.random.permutation(categorical_types).tolist()
        
        all_candidates = shuffled_non_cat + shuffled_cat
        
        for axis in all_candidates:
            if num_chosen >= top_n:
                break
            
            if axis in categorical_types:
                if categorical_chosen:
                    continue
                weight = 1.0
                categorical_chosen = True
                author_sliders[axis] = weight
                num_chosen += 1
                continue
            
            direction = np.random.choice([-1.0, 1.0])
            magnitude = np.random.uniform(0.5, 1.0)
            weight = direction * magnitude
            
            if mix_toward is False:
                weight *= -1.0
            
            author_sliders[axis] = weight
            num_chosen += 1
        
        sliders[author_id] = author_sliders
    
    return pd.Series(sliders)

def choose_sliders_styleremix(
    author_scores,
    normalized_scores,
    top_n_styles_to_change,
    score_std,
) -> pd.Series:
    mean_scores = normalized_scores.mean(axis=0)
    author_ids = author_scores.index.to_list()
    diff_scores = normalized_scores - mean_scores
    diff_scores_std = abs(diff_scores).std(axis=0)

    all_axes = author_scores.columns
    sliders = {}

    for author_id in author_ids:
        author_sliders = abs(diff_scores).loc[author_id].sort_values(ascending=False)
        candidate_axes = author_sliders.index.to_list()[:top_n_styles_to_change]
        candidate_sliders = {}

        for axis in candidate_axes:
            n_std = abs(diff_scores.loc[author_id][axis]) / score_std[axis] if score_std[axis] > 0 else 0
            sign = 1.0 if diff_scores.loc[author_id][axis] > 0 else -1.0
            weight = sign * std_to_weight(n_std)
            candidate_sliders[axis] = weight

        sliders[author_id] = candidate_sliders

    return pd.Series(sliders)