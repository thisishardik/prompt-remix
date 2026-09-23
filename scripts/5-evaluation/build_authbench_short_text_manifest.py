#!/usr/bin/env python3
"""Build reproducible AuthBench short-text cohorts for the PromptRemix paper."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ATTRIBUTOR = "Blablablab/multilingual-style-representation"
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
ABSOLUTE_BINS = [-np.inf, 5, 10, 20, 50, 128, np.inf]
ABSOLUTE_LABELS = ["under_5", "5_10", "10_20", "20_50", "50_128", "128_plus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authbench-dir",
        type=Path,
        default=Path("/gscratch/ifml1/hiatus/data/raw/authbench"),
    )
    parser.add_argument(
        "--sampled-dir",
        type=Path,
        default=Path("/gscratch/stf/hardiksr/hiatus/data/sampled/authbench"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("promptremix-paper/short-text"),
    )
    parser.add_argument(
        "--length-source",
        choices=["authbench", "attributor"],
        default="authbench",
        help=(
            "Use AuthBench's stored token_length for a fast preliminary analysis, "
            "or retokenize with the frozen paper attributor."
        ),
    )
    parser.add_argument("--tokenizer-batch-size", type=int, default=512)
    return parser.parse_args()


def author_id(value: object) -> str:
    if isinstance(value, list):
        return str(value[0])
    return str(value)


def load_sampled_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {
            str(row["documentID"])
            for row in json.load(handle)
            if row.get("documentID") is not None
        }


def attributor_lengths(
    texts: list[str],
    batch_size: int,
) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ATTRIBUTOR)
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )
        lengths.extend(int(length) for length in encoded["length"])
    return lengths


def load_candidate_profile_sizes(
    language: str,
    authbench_dir: Path,
) -> pd.Series:
    path = authbench_dir / language / f"authbench_{language}_candidates.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = pd.read_json(path, lines=True, encoding="utf-8")
    author_column = next(
        column
        for column in ("authorSetIDs", "authorIDs", "authorID", "author")
        if column in candidates.columns
    )
    candidate_authors = candidates[author_column].apply(author_id)
    return candidate_authors.value_counts()


def assign_quantile_bins(
    values: pd.Series,
    cutpoints: tuple[float, float, float],
) -> pd.Series:
    q25, q50, q75 = cutpoints
    return pd.Series(
        np.searchsorted([q25, q50, q75], values.to_numpy(), side="right") + 1,
        index=values.index,
        dtype="int64",
    )


def load_cell(
    language: str,
    genre: str,
    candidate_profile_sizes: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    query_path = (
        args.authbench_dir
        / language
        / "per_genre"
        / f"authbench_{language}_{genre}_queries.jsonl"
    )
    if not query_path.exists():
        raise FileNotFoundError(query_path)

    frame = pd.read_json(query_path, lines=True, encoding="utf-8")
    frame["index"] = frame.index
    author_column = next(
        column
        for column in ("authorSetIDs", "authorIDs", "authorID", "author")
        if column in frame.columns
    )
    frame["author_id"] = frame[author_column].apply(author_id)
    frame["language"] = language
    frame["primary_genre"] = genre

    if args.length_source == "attributor":
        frame["paper_token_length"] = attributor_lengths(
            frame["fullText"].fillna("").astype(str).tolist(),
            args.tokenizer_batch_size,
        )
    else:
        frame["paper_token_length"] = frame["token_length"].astype(int)

    sampled_path = (
        args.sampled_dir
        / language
        / genre
        / f"authbench_{language}_{genre}_sampled_ids.json"
    )
    sampled_ids = load_sampled_ids(sampled_path)
    frame["in_current_sample"] = frame["documentID"].astype(str).isin(sampled_ids)

    frame["candidate_profile_documents"] = (
        frame["author_id"].map(candidate_profile_sizes).fillna(0).astype(int)
    )
    for profile_size in (1, 2, 4):
        frame[f"candidate_profile_m{profile_size}_eligible"] = (
            frame["candidate_profile_documents"] >= profile_size
        )

    cell_cutpoints = tuple(
        float(value)
        for value in frame["paper_token_length"].quantile([0.25, 0.5, 0.75])
    )
    frame["cell_length_quartile"] = assign_quantile_bins(
        frame["paper_token_length"],
        cell_cutpoints,
    )
    frame["absolute_length_bin"] = pd.cut(
        frame["paper_token_length"],
        bins=ABSOLUTE_BINS,
        labels=ABSOLUTE_LABELS,
        right=False,
    ).astype(str)

    cell_p10 = float(frame["paper_token_length"].quantile(0.10))
    cell_p25 = float(frame["paper_token_length"].quantile(0.25))
    frame["cell_bottom_10pct"] = frame["paper_token_length"] <= cell_p10
    frame["cell_bottom_25pct"] = frame["paper_token_length"] <= cell_p25
    frame["cell_p10_cutpoint"] = cell_p10
    frame["cell_p25_cutpoint"] = cell_p25
    frame["cell_q50_cutpoint"] = cell_cutpoints[1]
    frame["cell_q75_cutpoint"] = cell_cutpoints[2]
    return frame


def describe_group(group: pd.DataFrame, scope: str) -> dict[str, object]:
    lengths = group["paper_token_length"]
    return {
        "scope": scope,
        "language": group["language"].iloc[0],
        "genre": (
            group["primary_genre"].iloc[0]
            if group["primary_genre"].nunique() == 1
            else "all"
        ),
        "n_queries": len(group),
        "n_authors": group["author_id"].nunique(),
        "n_current_sample": int(group["in_current_sample"].sum()),
        "mean_tokens": float(lengths.mean()),
        "std_tokens": float(lengths.std()),
        "min_tokens": int(lengths.min()),
        "p10_tokens": float(lengths.quantile(0.10)),
        "p25_tokens": float(lengths.quantile(0.25)),
        "median_tokens": float(lengths.median()),
        "p75_tokens": float(lengths.quantile(0.75)),
        "max_tokens": int(lengths.max()),
        "n_bottom_10pct": int(group["cell_bottom_10pct"].sum()),
        "n_bottom_25pct": int(group["cell_bottom_25pct"].sum()),
        "n_profile_m1": int(group["candidate_profile_m1_eligible"].sum()),
        "n_profile_m2": int(group["candidate_profile_m2_eligible"].sum()),
        "n_profile_m4": int(group["candidate_profile_m4_eligible"].sum()),
    }


def bin_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (language, genre, length_bin), group in frame.groupby(
        ["language", "primary_genre", "absolute_length_bin"],
        observed=True,
    ):
        rows.append(
            {
                "language": language,
                "genre": genre,
                "length_bin": length_bin,
                "n_queries": len(group),
                "n_authors": group["author_id"].nunique(),
                "n_current_sample": int(group["in_current_sample"].sum()),
                "n_profile_m1": int(
                    group["candidate_profile_m1_eligible"].sum()
                ),
                "n_profile_m2": int(
                    group["candidate_profile_m2_eligible"].sum()
                ),
                "n_profile_m4": int(
                    group["candidate_profile_m4_eligible"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_profile_sizes = {
        language: load_candidate_profile_sizes(language, args.authbench_dir)
        for language in GENRES
    }
    cells = [
        load_cell(language, genre, candidate_profile_sizes[language], args)
        for language, genres in GENRES.items()
        for genre in genres
    ]
    all_rows = pd.concat(cells, ignore_index=True)

    language_cutpoints = all_rows.groupby("language")["paper_token_length"].quantile(
        [0.10, 0.25]
    )
    all_rows["language_p10_cutpoint"] = all_rows["language"].map(
        {
            language: float(language_cutpoints.loc[(language, 0.10)])
            for language in GENRES
        }
    )
    all_rows["language_p25_cutpoint"] = all_rows["language"].map(
        {
            language: float(language_cutpoints.loc[(language, 0.25)])
            for language in GENRES
        }
    )
    all_rows["language_bottom_10pct"] = (
        all_rows["paper_token_length"] <= all_rows["language_p10_cutpoint"]
    )
    all_rows["language_bottom_25pct"] = (
        all_rows["paper_token_length"] <= all_rows["language_p25_cutpoint"]
    )

    manifest_columns = [
        "index",
        "documentID",
        "author_id",
        "language",
        "primary_genre",
        "source",
        "token_length",
        "paper_token_length",
        "absolute_length_bin",
        "cell_length_quartile",
        "cell_bottom_10pct",
        "cell_bottom_25pct",
        "language_bottom_10pct",
        "language_bottom_25pct",
        "in_current_sample",
        "candidate_profile_documents",
        "candidate_profile_m1_eligible",
        "candidate_profile_m2_eligible",
        "candidate_profile_m4_eligible",
    ]
    short_rows = all_rows[
        all_rows["cell_bottom_25pct"] | all_rows["language_bottom_25pct"]
    ]
    short_rows[manifest_columns].to_json(
        args.output_dir / "short_text_manifest.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    summary_rows = []
    for _, group in all_rows.groupby(["language", "primary_genre"]):
        summary_rows.append(describe_group(group, "language_genre"))
    for _, group in all_rows.groupby("language"):
        summary_rows.append(describe_group(group, "language"))
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "length_summary.csv",
        index=False,
    )
    bin_summary(all_rows).to_csv(
        args.output_dir / "absolute_bin_summary.csv",
        index=False,
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "length_source": args.length_source,
        "attributor": ATTRIBUTOR,
        "authbench_dir": str(args.authbench_dir.resolve()),
        "sampled_dir": str(args.sampled_dir.resolve()),
        "absolute_bins": ABSOLUTE_LABELS,
        "n_queries": len(all_rows),
        "n_short_manifest_queries": len(short_rows),
    }
    with (args.output_dir / "manifest_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
