#!/usr/bin/env python3
"""Summarize five seeded random-axis runs against round-robin PromptRemix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ATTRIBUTOR_TAG = "Blablablab_multilingual-style-representation"
SEEDS = (42, 43, 44, 45, 46)
GENRES = {
    "en": [
        "blog",
        "ecommerce_reviews",
        "literature",
        "news",
        "poetry",
        "qna",
        "research_paper",
    ],
    "ru": ["literature", "news", "poetry", "qna", "social_media"],
    "zh": [
        "ecommerce_reviews",
        "literature",
        "media_reviews",
        "news",
        "poetry",
        "social_media",
    ],
    "ar": ["literature", "news", "poetry", "social_media"],
}
SENSE_DIMENSIONS = [
    "semantic_coverage",
    "factuality",
    "informativeness",
    "relevance",
    "specificity",
    "correctness",
    "accuracy",
    "semantically_appropriate",
    "consistency",
    "coherence",
    "fluency",
    "quality",
    "understandability",
]


def sense_mean(path: Path) -> float:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scores = [row.get(field) for field in SENSE_DIMENSIONS]
            valid = [float(score) for score in scores if score is not None]
            if valid:
                values.append(mean(valid))
    if not values:
        raise ValueError(f"No valid sense scores in {path}")
    return mean(values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path(
            "/gscratch/stf/hardiksr/hiatus/metrics/"
            "paper_experiments/random_axis"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("promptremix-paper/analysis/random-axis"),
    )
    args = parser.parse_args()

    cell_rows = []
    control_cells = []
    for language, genres in GENRES.items():
        for genre in genres:
            control_suffix = f"promptremix_authbench-{language}0-3"
            control_privacy_path = (
                args.metrics_root
                / "privacy"
                / language
                / (
                    f"authbench_{language}_{genre}_{ATTRIBUTOR_TAG}_"
                    f"{control_suffix}.json"
                )
            )
            control_sense_path = (
                args.metrics_root
                / "sense"
                / language
                / f"{genre}_{control_suffix}_llm.jsonl"
            )
            if not control_privacy_path.exists() or not control_sense_path.exists():
                raise FileNotFoundError(
                    "Missing round-robin control result: "
                    f"{control_privacy_path} or {control_sense_path}"
                )
            control_privacy = json.loads(
                control_privacy_path.read_text(encoding="utf-8")
            )
            control_cells.append(
                {
                    "language": language,
                    "genre": genre,
                    "delta_eer": float(control_privacy["delta_eer"]),
                    "sense_13_mean_0_20": sense_mean(control_sense_path),
                }
            )
            for seed in SEEDS:
                suffix = (
                    f"promptremix_authbench-{language}0-3-random-seed{seed}"
                )
                privacy_path = (
                    args.metrics_root
                    / "privacy"
                    / language
                    / (
                        f"authbench_{language}_{genre}_{ATTRIBUTOR_TAG}_"
                        f"{suffix}.json"
                    )
                )
                sense_path = (
                    args.metrics_root
                    / "sense"
                    / language
                    / f"{genre}_{suffix}_llm.jsonl"
                )
                if not privacy_path.exists() or not sense_path.exists():
                    raise FileNotFoundError(
                        f"Missing seeded result: {privacy_path} or {sense_path}"
                    )
                privacy = json.loads(privacy_path.read_text(encoding="utf-8"))
                cell_rows.append(
                    {
                        "language": language,
                        "genre": genre,
                        "seed": seed,
                        "delta_eer": float(privacy["delta_eer"]),
                        "sense_13_mean_0_20": sense_mean(sense_path),
                    }
                )

    seed_groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in cell_rows:
        seed_groups[(str(row["language"]), int(row["seed"]))].append(row)
        seed_groups[("macro", int(row["seed"]))].append(row)
    seed_rows = []
    for (language, seed), rows in sorted(seed_groups.items()):
        seed_rows.append(
            {
                "language": language,
                "seed": seed,
                "genre_cells": len(rows),
                "delta_eer_macro_mean": mean(
                    float(row["delta_eer"]) for row in rows
                ),
                "sense_13_macro_mean_0_20": mean(
                    float(row["sense_13_mean_0_20"]) for row in rows
                ),
            }
        )

    language_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        language_groups[str(row["language"])].append(row)
    summary_rows = []
    control_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in control_cells:
        control_groups[str(row["language"])].append(row)
        control_groups["macro"].append(row)
    for language, rows in sorted(language_groups.items()):
        privacy = [float(row["delta_eer_macro_mean"]) for row in rows]
        utility = [float(row["sense_13_macro_mean_0_20"]) for row in rows]
        controls = control_groups[language]
        control_privacy = mean(
            float(row["delta_eer"]) for row in controls
        )
        control_utility = mean(
            float(row["sense_13_mean_0_20"]) for row in controls
        )
        summary_rows.append(
            {
                "language": language,
                "n_seeds": len(rows),
                "round_robin_delta_eer": control_privacy,
                "delta_eer_seed_mean": mean(privacy),
                "delta_eer_seed_std": stdev(privacy),
                "random_minus_round_robin_delta_eer": (
                    mean(privacy) - control_privacy
                ),
                "round_robin_sense_13_0_20": control_utility,
                "sense_13_seed_mean_0_20": mean(utility),
                "sense_13_seed_std": stdev(utility),
                "random_minus_round_robin_sense": (
                    mean(utility) - control_utility
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "random_axis_cells.csv", cell_rows)
    write_csv(args.output_dir / "round_robin_control_cells.csv", control_cells)
    write_csv(args.output_dir / "random_axis_by_seed.csv", seed_rows)
    write_csv(args.output_dir / "random_axis_mean_std.csv", summary_rows)
    print(f"Wrote random-axis summaries to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
