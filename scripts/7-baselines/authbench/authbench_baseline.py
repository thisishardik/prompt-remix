import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
import pandas as pd

from hiatus.evaluation.eer import (
    compute_delta_eer,
    compute_eer_from_embeddings,
)
from hiatus.evaluation.embedding_extractor import EmbeddingExtractor


def load_jsonl(filepath: str) -> Tuple[List[str], List[str], List[str]]:
    df = pd.read_json(filepath, lines=True, encoding="utf-8")

    doc_ids = df["documentID"].tolist()
    texts = df["fullText"].tolist()

    if "authorIDs" in df.columns:
        author_labels = (
            df["authorIDs"].apply(lambda x: x[0] if isinstance(x, list) else x).tolist()
        )
    elif "authorSetIDs" in df.columns:
        author_labels = (
            df["authorSetIDs"]
            .apply(lambda x: x[0] if isinstance(x, list) else x)
            .tolist()
        )
    else:
        raise ValueError(
            f"JSONL file {filepath} must contain 'authorIDs' or 'authorSetIDs' column. "
            f"Found columns: {list(df.columns)}"
        )

    print(
        f"Loaded {len(texts)} documents from {filepath}"
    )
    return doc_ids, texts, author_labels


def run_evaluation(
    model_name: str,
    queries_jsonl: str,
    candidates_jsonl: str,
    ground_truth_file: str,
    privatized_queries_jsonl: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 32,
    use_flash_attention: bool = False,
    quantize_4bit: bool = False,
) -> Dict:

    q_ids, q_texts, q_labels = load_jsonl(queries_jsonl)
    c_ids, c_texts, c_labels = load_jsonl(candidates_jsonl)

    priv_ids, priv_texts, priv_labels = None, None, None
    if privatized_queries_jsonl:
        priv_ids, priv_texts, priv_labels = load_jsonl(privatized_queries_jsonl)

    ground_truth_pairs = np.load(ground_truth_file).astype(int)
    ground_truth_pairs = ground_truth_pairs.flatten()
    print(f"Loaded {len(ground_truth_pairs)} ground truth pairs from {ground_truth_file}")

    extractor = EmbeddingExtractor(
        model_name,
        use_flash_attention=use_flash_attention,
        quantize_4bit=quantize_4bit,
    )

    print("Encoding queries...")
    q_embs = extractor.encode(q_texts, batch_size=batch_size)

    print("Encoding candidates...")
    c_embs = extractor.encode(c_texts, batch_size=batch_size)

    priv_embs = None
    if priv_texts:
        print("Encoding privatized queries...")
        priv_embs = extractor.encode(priv_texts, batch_size=batch_size)

    extractor.unload()

    print("Computing EER for original queries...")
    original_results = compute_eer_from_embeddings(
        query_embeddings=q_embs,
        candidate_embeddings=c_embs,
        ground_truth_pairs=ground_truth_pairs,
        query_labels=q_labels,
        candidate_labels=c_labels
    )
    print(f"Original EER: {original_results['eer']:.4f}")
    print(f"Threshold: {original_results['threshold']:.4f}")
    print(f"Positive pairs: {original_results['n_positive_pairs']}")
    print(f"Negative pairs: {original_results['n_negative_pairs']}")

    results = {
        "model": model_name,
        "eer_original": original_results["eer"],
        "threshold_original": original_results["threshold"],
        "n_positive_pairs": original_results["n_positive_pairs"],
        "n_negative_pairs": original_results["n_negative_pairs"],
        "n_queries": len(q_texts),
        "n_candidates": len(c_texts)
    }

    if priv_embs is not None:
        print("Computing EER for privatized queries...")
        privatized_results = compute_eer_from_embeddings(
            query_embeddings=priv_embs,
            candidate_embeddings=c_embs,
            ground_truth_pairs=ground_truth_pairs,
            query_labels=priv_labels,
            candidate_labels=c_labels
        )
        print(f"  Privatized EER: {privatized_results['eer']:.4f}")

        delta_eer = compute_delta_eer(
            original_results["eer"], privatized_results["eer"]
        )
        print(f"  Delta EER: {delta_eer:+.4f}")

        results.update(
            {
                "eer_privatized": privatized_results["eer"],
                "threshold_privatized": privatized_results["threshold"],
                "delta_eer": delta_eer,
            }
        )

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = (
            output_path
            / f"authbench_eer_{model_name.replace('/', '_')}.json"
        )
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_file}")

        emb_file = (
            output_path
            / f"embeddings_{model_name.replace('/', '_')}.npz"
        )
        save_dict = {
            "query_embeddings": q_embs,
            "candidate_embeddings": c_embs,
            "query_labels": np.array(q_labels),
            "candidate_labels": np.array(c_labels),
        }

        if priv_embs is not None:
            save_dict["privatized_query_embeddings"] = priv_embs
        np.savez_compressed(emb_file, **save_dict)
        print(f"Embeddings saved to {emb_file}")

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True
    )
    parser.add_argument(
        "--queries-jsonl",
        type=str,
        required=True
    )
    parser.add_argument(
        "--candidates-jsonl",
        type=str,
        required=True
    )
    parser.add_argument(
        "--privatized-queries-jsonl",
        type=str,
        default=None
    )
    parser.add_argument("--ground-truth-file", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true"
    )
    parser.add_argument(
        "--quantize-4bit",
        action="store_true"
    )

    args = parser.parse_args()

    print(args)

    results = run_evaluation(
        model_name=args.model,
        queries_jsonl=args.queries_jsonl,
        candidates_jsonl=args.candidates_jsonl,
        privatized_queries_jsonl=args.privatized_queries_jsonl,
        ground_truth_file=args.ground_truth_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_flash_attention=args.flash_attention,
        quantize_4bit=args.quantize_4bit,
    )


if __name__ == "__main__":
    main()
