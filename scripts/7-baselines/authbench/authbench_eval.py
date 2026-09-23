import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from hiatus.data import authbench_paths, blog_authorship_paths, hrs_paths
from hiatus.evaluation.eer import (
    compute_delta_eer,
    compute_eer,
    construct_verification_pairs,
)
from hiatus.evaluation.embedding_extractor import EmbeddingExtractor


def load_authbench_jsonl(filepath: str) -> Tuple[List[str], List[str], List[str]]:
    df = pd.read_json(filepath, lines=True, encoding="utf-8")

    doc_ids = df["documentID"].tolist()
    texts = df["fullText"].tolist()

    author_key = next((k for k in ["authorSetIDs", "authorIDs", "authorID", "author"] if k in df.columns), None)
    if not author_key:
        raise KeyError(f"Could not find author key in columns: {df.columns}")

    author_labels = (
        df[author_key].apply(lambda x: x[0] if isinstance(x, list) else x).tolist()
    )

    print(f"Loaded {len(texts)} documents from {filepath}")
    return doc_ids, texts, author_labels


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    ground_truth_matrix: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    q_norm = query_embeddings / (
        np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12
    )
    c_norm = candidate_embeddings / (
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-12
    )

    similarity = q_norm @ c_norm.T  # (n_queries, n_candidates)

    n_queries = similarity.shape[0]
    success_at_k = 0.0
    recall_at_k = 0.0
    ndcg_at_k = 0.0

    for i in range(n_queries):
        scores = similarity[i]
        top_k_indices = np.argsort(scores)[::-1][:k]

        rel = ground_truth_matrix[i, top_k_indices]
        n_pos = ground_truth_matrix[i].sum()

        if rel.sum() > 0:
            success_at_k += 1.0

        if n_pos > 0:
            recall_at_k += rel.sum() / n_pos

        dcg = sum(rel[j] / np.log2(j + 2) for j in range(len(rel)))
        ideal_rel = np.sort(ground_truth_matrix[i])[::-1][:k]
        idcg = sum(ideal_rel[j] / np.log2(j + 2) for j in range(len(ideal_rel)))
        if idcg > 0:
            ndcg_at_k += dcg / idcg

    return {
        f"success_at_{k}": success_at_k / n_queries,
        f"recall_at_{k}": recall_at_k / n_queries,
        f"ndcg_at_{k}": ndcg_at_k / n_queries,
    }


def run_evaluation(
    model_name: str,
    language: str,
    dataset: str = "authbench",
    sample_name: Optional[str] = None,
    genre: Optional[str] = None,
    pipeline_suffix: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 32,
    use_flash_attention: bool = False,
    quantize_4bit: bool = False,
    compute_retrieval: bool = True,
    k: int = 5,
    sampled_data_filepath: Optional[str] = None,
) -> Dict:

    if dataset == "authbench":
        paths = authbench_paths(
            language=language,
            genre=genre,
            pipeline_suffix=(
                pipeline_suffix
                or f"promptremix_authbench-{language}0-3"
            ),
        )
    elif dataset == "blog":
        paths = blog_authorship_paths(
            sample_name=sample_name,
            pipeline_suffix=pipeline_suffix,
        )
    elif dataset == "hrs":
        hrs_dict = hrs_paths(
            genre=genre,
            language=language,
            pipeline_suffix=pipeline_suffix or "mutantx",
        )["TA3"]
        paths = {
            "queries": hrs_dict["data"]["queries"],
            "candidates": hrs_dict["data"]["candidates"],
            "privatized": hrs_dict["data"]["privatized"],
            "ground_truth": hrs_dict["groundtruth"]["groundtruth"],
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    queries_path = str(paths["queries"])
    candidates_path = str(paths["candidates"])
    gt_path = str(paths["ground_truth"])
    privatized_path = str(paths["privatized"])

    q_ids, q_texts, q_labels = load_authbench_jsonl(queries_path)
    c_ids, c_texts, c_labels = load_authbench_jsonl(candidates_path)

    gt_matrix = np.load(gt_path).astype(int)

    has_privatized = pipeline_suffix and Path(privatized_path).exists()
    priv_ids, priv_texts, priv_labels = None, None, None
    if has_privatized:
        priv_ids, priv_texts, priv_labels = load_authbench_jsonl(privatized_path)
    else:
        print(f"Privatized queries not found at {privatized_path}")

    if sampled_data_filepath and Path(sampled_data_filepath).exists():
        with open(sampled_data_filepath, "r") as f:
            sampled_docs = json.load(f)
        sampled_doc_ids = {doc["documentID"] for doc in sampled_docs}
        
        valid_indices = [i for i, doc_id in enumerate(q_ids) if doc_id in sampled_doc_ids]
        q_ids = [q_ids[i] for i in valid_indices]
        q_texts = [q_texts[i] for i in valid_indices]
        q_labels = [q_labels[i] for i in valid_indices]
        gt_matrix = gt_matrix[valid_indices, :]
        print(f"Filtered to {len(q_ids)} sampled queries.")
        
        if priv_ids is not None:
            priv_valid_indices = [i for i, doc_id in enumerate(priv_ids) if doc_id in sampled_doc_ids]
            priv_ids = [priv_ids[i] for i in priv_valid_indices]
            priv_texts = [priv_texts[i] for i in priv_valid_indices]
            priv_labels = [priv_labels[i] for i in priv_valid_indices]

    gt_flat = gt_matrix.flatten()
    print(f"Ground truth shape: {gt_matrix.shape}, positive pairs: {gt_matrix.sum()}")

    if dataset == "hrs":
        print("Collapsing documents to authors for HRS dataset...")
        with open(hrs_dict["groundtruth"]["query_labels"], "r") as f:
            true_q_labels = [eval(line.strip())[0] for line in f]
        with open(hrs_dict["groundtruth"]["candidate_labels"], "r") as f:
            true_c_labels = [eval(line.strip())[0] for line in f]
            
        q_df = pd.DataFrame({"text": q_texts, "label": q_labels})
        q_grouped = q_df.groupby("label")["text"].apply(lambda x: " ".join(x)).to_dict()
        q_texts = [q_grouped.get(l, "") for l in true_q_labels]
        q_labels = true_q_labels
        
        c_df = pd.DataFrame({"text": c_texts, "label": c_labels})
        c_grouped = c_df.groupby("label")["text"].apply(lambda x: " ".join(x)).to_dict()
        c_texts = [c_grouped.get(l, "") for l in true_c_labels]
        c_labels = true_c_labels
        
        if priv_texts is not None:
            p_df = pd.DataFrame({"text": priv_texts, "label": priv_labels})
            p_grouped = p_df.groupby("label")["text"].apply(lambda x: " ".join(x)).to_dict()
            priv_texts = [p_grouped.get(l, "") for l in true_q_labels]
            priv_labels = true_q_labels

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
    if has_privatized:
        print("Encoding privatized queries...")
        priv_embs = extractor.encode(priv_texts, batch_size=batch_size)

    extractor.unload()

    print("Computing EER for original queries...")
    scores_pos, scores_neg = construct_verification_pairs(
        query_embeddings=q_embs,
        candidate_embeddings=c_embs,
        ground_truth_pairs=gt_flat,
        query_labels=q_labels,
        candidate_labels=c_labels,
    )
    eer_orig, thresh_orig = compute_eer(scores_pos, scores_neg)
    print(f"Original EER: {eer_orig:.4f} (threshold: {thresh_orig:.4f})")
    print(f"Positive pairs: {len(scores_pos)}, Negative pairs: {len(scores_neg)}")

    results = {
        "model": model_name,
        "language": language,
        "dataset": dataset,
        "sample_name": sample_name,
        "genre": genre or "all",
        "eer_original": eer_orig,
        "threshold_original": thresh_orig,
        "n_queries": len(q_texts),
        "n_candidates": len(c_texts),
        "n_positive_pairs": len(scores_pos),
        "n_negative_pairs": len(scores_neg),
    }

    if compute_retrieval:
        print(f"Computing retrieval metrics (K={k})...")
        retrieval = compute_retrieval_metrics(q_embs, c_embs, gt_matrix, k=k)
        print(f"Success@{k}: {retrieval[f'success_at_{k}']:.4f}")
        print(f"Recall@{k}: {retrieval[f'recall_at_{k}']:.4f}")
        print(f"nDCG@{k}: {retrieval[f'ndcg_at_{k}']:.4f}")
        results.update(retrieval)

    if priv_embs is not None:
        print("Computing EER for privatized queries...")
        scores_pos_p, scores_neg_p = construct_verification_pairs(
            query_embeddings=priv_embs,
            candidate_embeddings=c_embs,
            ground_truth_pairs=gt_flat,
            query_labels=priv_labels,
            candidate_labels=c_labels,
        )
        eer_priv, thresh_priv = compute_eer(scores_pos_p, scores_neg_p)
        delta = compute_delta_eer(eer_orig, eer_priv)
        print(f"Privatized EER: {eer_priv:.4f}")
        print(f"Delta EER: {delta:+.4f}")

        results.update({
            "eer_privatized": eer_priv,
            "threshold_privatized": thresh_priv,
            "delta_eer": delta,
            "pipeline_suffix": pipeline_suffix,
        })

        if compute_retrieval:
            print(f"Computing retrieval metrics for privatized queries (K={k})...")
            retrieval_priv = compute_retrieval_metrics(
                priv_embs, c_embs, gt_matrix, k=k
            )
            for key, val in retrieval_priv.items():
                results[f"{key}_privatized"] = val
            print(f"Success@{k} (priv): {retrieval_priv[f'success_at_{k}']:.4f}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        genre_tag = f"_{genre.replace('/', '_')}" if genre else ""
        sample_tag = f"_{sample_name}" if sample_name else ""
        model_tag = model_name.replace("/", "_")
        suffix_tag = f"_{pipeline_suffix}" if pipeline_suffix else ""

        results_file = (
            out / f"{dataset}_{language}{genre_tag}{sample_tag}_{model_tag}{suffix_tag}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")

        emb_file = out / (
            f"embeddings_{dataset}_{language}{genre_tag}{sample_tag}_"
            f"{model_tag}{suffix_tag}.npz"
        )
        save_dict = {
            "query_embeddings": q_embs,
            "candidate_embeddings": c_embs,
            "query_ids": np.asarray(q_ids),
            "query_labels": np.asarray(q_labels),
            "candidate_labels": np.asarray(c_labels),
        }
        if priv_embs is not None:
            save_dict["privatized_query_embeddings"] = priv_embs
        np.savez_compressed(emb_file, **save_dict)
        print(f"Embeddings saved to {emb_file}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--language", type=str, required=True, choices=["en", "ru", "zh", "ar"]
    )
    parser.add_argument("--dataset", type=str, default="authbench", choices=["authbench", "blog", "hrs"])
    parser.add_argument("--sample-name", type=str, default=None)
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--pipeline-suffix", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--quantize-4bit", action="store_true")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sampled-data-filepath", type=str, default=None)

    args = parser.parse_args()
    print(args)

    results = run_evaluation(
        model_name=args.model,
        language=args.language,
        dataset=args.dataset,
        sample_name=args.sample_name,
        genre=args.genre,
        pipeline_suffix=args.pipeline_suffix,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_flash_attention=args.flash_attention,
        quantize_4bit=args.quantize_4bit,
        compute_retrieval=not args.no_retrieval,
        k=args.k,
        sampled_data_filepath=args.sampled_data_filepath,
    )


if __name__ == "__main__":
    main()
