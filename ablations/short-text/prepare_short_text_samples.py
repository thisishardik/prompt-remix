#!/usr/bin/env python3
"""Create deterministic, author-stratified AuthBench short-text samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("promptremix-paper/short-text/short_text_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/gscratch/stf/hardiksr/hiatus/data/sampled/"
            "authbench_short_text/bottom25"
        ),
    )
    parser.add_argument("--authors-per-cell", type=int, default=100)
    parser.add_argument("--max-docs-per-author", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_cell(
    frame: pd.DataFrame,
    authors_per_cell: int,
    max_docs_per_author: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    authors = sorted(frame["author_id"].unique().tolist())
    chosen_authors = rng.sample(authors, min(authors_per_cell, len(authors)))
    rows: list[dict[str, object]] = []
    for author in chosen_authors:
        author_rows = frame[frame["author_id"] == author]
        chosen_indices = rng.sample(
            author_rows.index.tolist(),
            min(max_docs_per_author, len(author_rows)),
        )
        for index in chosen_indices:
            row = author_rows.loc[index]
            rows.append(
                {
                    "index": int(row["index"]),
                    "documentID": str(row["documentID"]),
                    "paper_token_length": int(row["paper_token_length"]),
                    "absolute_length_bin": str(row["absolute_length_bin"]),
                    "cell_bottom_10pct": bool(row["cell_bottom_10pct"]),
                    "cell_bottom_25pct": bool(row["cell_bottom_25pct"]),
                    "author_id": str(row["author_id"]),
                }
            )
    return sorted(rows, key=lambda row: int(row["index"]))


def main() -> None:
    args = parse_args()
    frame = pd.read_json(args.manifest, lines=True)
    frame = frame[frame["cell_bottom_25pct"]].copy()

    summary = []
    for language, genres in GENRES.items():
        for genre_index, genre in enumerate(genres):
            cell = frame[
                (frame["language"] == language)
                & (frame["primary_genre"] == genre)
            ]
            rows = sample_cell(
                cell,
                args.authors_per_cell,
                args.max_docs_per_author,
                args.seed + genre_index,
            )
            output_dir = args.output_root / language / genre
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir
                / f"authbench_{language}_{genre}_sampled_ids.json"
            )
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(rows, handle, ensure_ascii=False, indent=2)
            summary.append(
                {
                    "language": language,
                    "genre": genre,
                    "available_queries": len(cell),
                    "available_authors": cell["author_id"].nunique(),
                    "sampled_queries": len(rows),
                    "sampled_authors": len({row["author_id"] for row in rows}),
                    "output_path": str(output_path),
                }
            )

    summary_path = args.output_root / "sample_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
