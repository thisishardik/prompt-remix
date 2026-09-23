#!/usr/bin/env python3
"""Paired query-author bootstrap confidence intervals for AuthBench Delta EER."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from hiatus.evaluation.eer import compute_eer


ORIGINAL_SIMILARITY: np.ndarray
PRIVATIZED_SIMILARITY: np.ndarray
GROUND_TRUTH: np.ndarray
AUTHOR_ROWS: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def eer_for_rows(similarity: np.ndarray, rows: np.ndarray) -> float:
    selected_scores = similarity[rows].reshape(-1)
    selected_truth = GROUND_TRUTH[rows].reshape(-1)
    eer, _ = compute_eer(
        selected_scores[selected_truth],
        selected_scores[~selected_truth],
    )
    return eer


def bootstrap_once(seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    sampled_author_indices = rng.integers(0, len(AUTHOR_ROWS), len(AUTHOR_ROWS))
    rows = np.concatenate([AUTHOR_ROWS[index] for index in sampled_author_indices])
    original = eer_for_rows(ORIGINAL_SIMILARITY, rows)
    privatized = eer_for_rows(PRIVATIZED_SIMILARITY, rows)
    return original, privatized, privatized - original


def interval(values: np.ndarray, confidence: float) -> list[float]:
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    ]


def main() -> None:
    args = parse_args()
    artifact = np.load(args.embeddings)
    required = {
        "query_embeddings",
        "candidate_embeddings",
        "privatized_query_embeddings",
        "query_labels",
        "candidate_labels",
    }
    missing = required.difference(artifact.files)
    if missing:
        raise KeyError(
            f"{args.embeddings} lacks bootstrap metadata: {sorted(missing)}. "
            "Rerun authbench_eval.py with the updated artifact format."
        )

    query_embeddings = artifact["query_embeddings"]
    candidate_embeddings = artifact["candidate_embeddings"]
    privatized_embeddings = artifact["privatized_query_embeddings"]
    query_labels = artifact["query_labels"].astype(str)
    candidate_labels = artifact["candidate_labels"].astype(str)

    global ORIGINAL_SIMILARITY
    global PRIVATIZED_SIMILARITY
    global GROUND_TRUTH
    global AUTHOR_ROWS
    candidate_normalized = normalize(candidate_embeddings)
    ORIGINAL_SIMILARITY = normalize(query_embeddings) @ candidate_normalized.T
    PRIVATIZED_SIMILARITY = (
        normalize(privatized_embeddings) @ candidate_normalized.T
    )
    GROUND_TRUTH = query_labels[:, None] == candidate_labels[None, :]
    authors = np.unique(query_labels)
    AUTHOR_ROWS = [np.flatnonzero(query_labels == author) for author in authors]

    all_rows = np.arange(len(query_labels))
    point_original = eer_for_rows(ORIGINAL_SIMILARITY, all_rows)
    point_privatized = eer_for_rows(PRIVATIZED_SIMILARITY, all_rows)

    seeds = np.random.SeedSequence(args.seed).generate_state(args.n_bootstrap)
    if args.workers == 1:
        replicates = [bootstrap_once(int(seed)) for seed in seeds]
    else:
        with mp.get_context("fork").Pool(args.workers) as pool:
            replicates = pool.map(bootstrap_once, (int(seed) for seed in seeds))
    values = np.asarray(replicates)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["replicate", "eer_original", "eer_privatized", "delta_eer"]
        )
        for index, row in enumerate(values):
            writer.writerow([index, *row.tolist()])

    confidence = args.confidence
    summary = {
        "embeddings": str(args.embeddings.resolve()),
        "bootstrap_unit": "query_author",
        "paired": True,
        "n_queries": len(query_labels),
        "n_authors": len(authors),
        "n_candidates": len(candidate_labels),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "confidence": confidence,
        "eer_original": point_original,
        "eer_original_ci": interval(values[:, 0], confidence),
        "eer_privatized": point_privatized,
        "eer_privatized_ci": interval(values[:, 1], confidence),
        "delta_eer": point_privatized - point_original,
        "delta_eer_ci": interval(values[:, 2], confidence),
        "replicates_path": str(csv_path.resolve()),
    }
    json_path = args.output_prefix.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
