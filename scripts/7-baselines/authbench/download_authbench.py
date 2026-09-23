import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from datasets import load_dataset


TARGET_LANGUAGES = {"en", "ru", "zh", "ar"}


def download_and_filter(
    config: str,
    split: str,
    languages: Set[str],
) -> pd.DataFrame:
    print(f"Downloading AuthBench config='{config}', split='{split}'...")
    ds = load_dataset("MaoXun/AuthBench", config, split=split)
    df = ds.to_pandas()
    print(f"  Total records: {len(df)}")

    lang_col = "lang"
    if lang_col in df.columns:
        df = df[df[lang_col].isin(languages)].reset_index(drop=True)
        print(f"  After language filter ({languages}): {len(df)}")
    else:
        print(f"  Warning: no 'lang' column in {config}. Keeping all records.")
    return df


def convert_queries_to_hrs(df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    gt_author_map = dict(zip(gt_df["query_id"], gt_df["author_id"]))

    hrs_records = []
    for _, row in df.iterrows():
        query_id = row["query_id"]
        author_id = gt_author_map.get(query_id, "unknown")
        hrs_records.append({
            "documentID": query_id,
            "authorIDs": [author_id],
            "fullText": row["content"],
            "lang": row.get("lang", ""),
            "genre": row.get("genre", ""),
            "source": row.get("source", ""),
            "token_length": row.get("token_length", 0),
        })

    return pd.DataFrame(hrs_records)


def convert_candidates_to_hrs(df: pd.DataFrame) -> pd.DataFrame:
    hrs_records = []
    for _, row in df.iterrows():
        hrs_records.append({
            "documentID": row["candidate_id"],
            "authorIDs": [row["author_id"]],
            "fullText": row["content"],
            "lang": row.get("lang", ""),
            "genre": row.get("genre", ""),
            "source": row.get("source", ""),
            "token_length": row.get("token_length", 0),
        })

    return pd.DataFrame(hrs_records)


def build_ground_truth_matrix(
    queries_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> np.ndarray:
    query_ids = queries_df["documentID"].tolist()
    candidate_ids = candidates_df["documentID"].tolist()

    query_id_to_idx = {qid: i for i, qid in enumerate(query_ids)}
    candidate_id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

    n_q = len(query_ids)
    n_c = len(candidate_ids)
    gt_matrix = np.zeros((n_q, n_c), dtype=np.int32)

    for _, row in gt_df.iterrows():
        qid = row["query_id"]
        positive_ids = row["positive_ids"]

        if qid not in query_id_to_idx:
            continue

        q_idx = query_id_to_idx[qid]
        for pid in positive_ids:
            if pid in candidate_id_to_idx:
                c_idx = candidate_id_to_idx[pid]
                gt_matrix[q_idx, c_idx] = 1

    return gt_matrix


def save_per_language(
    queries_hrs: pd.DataFrame,
    candidates_hrs: pd.DataFrame,
    gt_df: pd.DataFrame,
    output_dir: Path,
    languages: Set[str],
) -> Dict[str, Dict]:
    stats = {}

    for lang in sorted(languages):
        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        q_mask = queries_hrs["lang"] == lang
        q_lang = queries_hrs[q_mask].reset_index(drop=True)

        c_mask = candidates_hrs["lang"] == lang
        c_lang = candidates_hrs[c_mask].reset_index(drop=True)

        if len(q_lang) == 0 or len(c_lang) == 0:
            print(f"  Skipping {lang}: {len(q_lang)} queries, {len(c_lang)} candidates")
            continue

        gt_lang = build_ground_truth_matrix(q_lang, c_lang, gt_df)

        q_path = lang_dir / f"authbench_{lang}_queries.jsonl"
        q_lang.to_json(q_path, orient="records", lines=True, force_ascii=False)

        c_path = lang_dir / f"authbench_{lang}_candidates.jsonl"
        c_lang.to_json(c_path, orient="records", lines=True, force_ascii=False)

        gt_path = lang_dir / f"authbench_{lang}_ground_truth.npy"
        np.save(gt_path, gt_lang)

        q_lang["top_genre"] = q_lang["genre"].apply(lambda x: x.split("/")[0])
        genres = sorted(q_lang["top_genre"].unique())
        
        genre_dir = lang_dir / "per_genre"
        genre_dir.mkdir(parents=True, exist_ok=True)

        genre_stats = {}
        for genre in genres:
            genre_safe = genre.replace("/", "_")

            gq_mask = q_lang["top_genre"] == genre
            gq = q_lang[gq_mask].reset_index(drop=True)

            gc = c_lang

            if len(gq) == 0:
                continue

            gt_genre = build_ground_truth_matrix(gq, gc, gt_df)
            gq_path = genre_dir / f"authbench_{lang}_{genre_safe}_queries.jsonl"
            gq.to_json(gq_path, orient="records", lines=True, force_ascii=False)
            gt_genre_path = genre_dir / f"authbench_{lang}_{genre_safe}_ground_truth.npy"
            np.save(gt_genre_path, gt_genre)

            genre_stats[genre] = {
                "n_queries": len(gq),
                "n_positive_pairs": int(gt_genre.sum()),
            }


        n_pos = int(gt_lang.sum())
        n_neg = int(gt_lang.size - n_pos)

        stats[lang] = {
            "n_queries": len(q_lang),
            "n_candidates": len(c_lang),
            "n_positive_pairs": n_pos,
            "n_negative_pairs": n_neg,
            "n_genres": len(genres),
            "genres": genres,
            "genre_stats": genre_stats,
            "queries_path": str(q_path),
            "candidates_path": str(c_path),
            "ground_truth_path": str(gt_path),
        }

        print(f"{lang}: {len(q_lang)} queries, {len(c_lang)} candidates, "
              f"{n_pos} positive pairs, {len(genres)} genres")
        for g, gs in genre_stats.items():
            print(f"{g}: {gs['n_queries']} queries, {gs['n_positive_pairs']} pos pairs")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/gscratch/ifml1/hiatus/data/authbench"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "ru", "zh", "ar"]
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    languages = set(args.languages)
    splits = ["train", "dev", "test"]

    print(f"Output directory: {output_dir}")
    print(f"Languages: {sorted(languages)}")
    print()

    all_queries = []
    all_candidates = []
    all_gt = []

    for split in splits:
        print(f"Downloading split: {split}")

        queries_raw = download_and_filter("queries", split, languages)
        candidates_raw = download_and_filter("candidates", split, languages)
        gt_raw = download_and_filter("ground_truth", split, languages)

        all_queries.append(queries_raw)
        all_candidates.append(candidates_raw)
        all_gt.append(gt_raw)

    print("Merging all splits...")

    queries_merged = pd.concat(all_queries, ignore_index=True)
    candidates_merged = pd.concat(all_candidates, ignore_index=True)
    gt_merged = pd.concat(all_gt, ignore_index=True)

    n_before = len(candidates_merged)
    candidates_merged = candidates_merged.drop_duplicates(subset=["candidate_id"]).reset_index(drop=True)
    print(f"Candidates: {n_before} -> {len(candidates_merged)} after dedup")

    n_before = len(queries_merged)
    queries_merged = queries_merged.drop_duplicates(subset=["query_id"]).reset_index(drop=True)
    print(f"Queries: {n_before} -> {len(queries_merged)} after dedup")

    n_before = len(gt_merged)
    gt_merged = gt_merged.drop_duplicates(subset=["query_id"]).reset_index(drop=True)
    print(f"Ground truth: {n_before} -> {len(gt_merged)} after dedup")

    print(f"\nMerged totals: {len(queries_merged)} queries, "
          f"{len(candidates_merged)} candidates, {len(gt_merged)} GT rows")

    print("\nConverting to HRS format...")
    queries_hrs = convert_queries_to_hrs(queries_merged, gt_merged)
    candidates_hrs = convert_candidates_to_hrs(candidates_merged)

    print(f"Queries (HRS): {len(queries_hrs)}")
    print(f"Candidates (HRS): {len(candidates_hrs)}")

    print("\nSaving per-language data...")
    stats = save_per_language(
        queries_hrs, candidates_hrs, gt_merged,
        output_dir, languages,
    )

    stats_path = output_dir / "download_stats.json"
    serializable_stats = {}
    for lang, s in stats.items():
        s_copy = {k: v for k, v in s.items()}
        s_copy.pop("genres", None)
        serializable_stats[lang] = s_copy

    with open(stats_path, "w") as f:
        json.dump(serializable_stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")

    print("Download and conversion complete!")


if __name__ == "__main__":
    main()

