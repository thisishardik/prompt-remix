### IMPORTANT: this should be run with the hiatus-eval environment.
import os
import argparse
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics.pairwise import cosine_similarity
import transformers

from hiatus.hrs import hrs_paths, authbench_paths
from hiatus.config import code_dir, metrics_dir
from hiatus.privacy.experiment import Experiment


####################################
# Embedding extraction
####################################
def extract_embeddings(
    model,
    filename: str,
    sampled_doc_ids: list = None,
    is_author_level: bool = False,
    is_author_emb_model: bool = True,
    batch_size: int = 32,
) -> Dataset:
    """
    Extract document- or author-level embeddings from a JSONL file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    data = pd.read_json(filename, lines=True, encoding="utf-8")
    
    if sampled_doc_ids is not None and len(sampled_doc_ids) > 0:
        data = data[data['documentID'].isin(sampled_doc_ids)].reset_index(drop=True)

    if is_author_level:
        # Collapse texts per author
        id_col = "authorIDs" if "authorIDs" in data.columns else "authorSetIDs"
        data[id_col] = data[id_col].apply(lambda x: x[0])
        grouped = data.groupby(id_col)["fullText"].apply(list).reset_index()
        all_ids, all_embs = [], []

        for _, row in grouped.iterrows():
            author_texts = row["fullText"]
            emb = (
                model.encode_single_author(author_texts)
                if is_author_emb_model
                else model.encode(author_texts).mean(dim=0)
            )
            all_ids.append(row[id_col])
            all_embs.append(emb.cpu().float().numpy().tolist())

        dataset = Dataset.from_dict({id_col: all_ids, "features": all_embs})
        return dataset

    else:
        # Document-level embeddings
        all_ids, all_embs, all_texts = [], [], []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i : i + batch_size]
            texts = batch["fullText"].tolist()
            embs = model.encode(texts)
            all_embs.extend(embs.cpu().float().numpy().tolist())
            all_ids.extend(batch["documentID"])
            all_texts.extend(texts)

        dataset = Dataset.from_dict(
            {"documentID": all_ids, "features": all_embs, "inputs": all_texts}
        )
        return dataset


####################################
# Similarity Scores
####################################
def compute_ta2_scores(
    dataset_path: Path,
    output_path: Path,
    query_split: Literal["queries", "privatized_queries"] = "queries",
) -> Dict[str, str]:
    """
    Load a DatasetDict with 'queries'/'privatized_queries' and 'candidates' and compute cosine similarity.
    Saves scores and label files, returning their paths.
    """
    dataset = DatasetDict.load_from_disk(str(dataset_path))
    query_feats = np.array(dataset[query_split]["features"])
    cand_feats = np.array(dataset["candidates"]["features"])

    query_labels = dataset[query_split][list(dataset[query_split].features.keys())[0]]
    cand_labels = dataset["candidates"][list(dataset["candidates"].features.keys())[0]]

    scores = cosine_similarity(query_feats, cand_feats)

    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "scores.npy", scores)

    q_labels_path = output_path / "query_labels.txt"
    c_labels_path = output_path / "candidate_labels.txt"

    with open(q_labels_path, "w") as fq:
        for q in query_labels:
            fq.write(f"{(q,)}\n")

    with open(c_labels_path, "w") as fc:
        for c in cand_labels:
            fc.write(f"{(c,)}\n")

    return {
        "scores": str(output_path / "scores.npy"),
        "query_labels": str(q_labels_path),
        "candidate_labels": str(c_labels_path),
    }


####################################
# Generate Experiment Configuration
####################################
def generate_evaluation_config(
    paths: Dict[str, Any], pipeline_suffix: str, gt_path: str, gt_q_labels_path: str, gt_c_labels_path: str
) -> Dict[str, Any]:
    """Build a minimal TA3 evaluation configuration dictionary."""
    return {
        "name": f"{pipeline_suffix}_evaluation",
        "ta3": {
            "privacy": {
                "metric_name": "Delta Equal Error Rate",
                "ground_truth": gt_path,
                "ground_truth_candidate_labels": gt_c_labels_path,
                "ground_truth_query_labels": gt_q_labels_path,
                "ta2_system_outputs": [
                    {
                        "ta2_system_name": "uoregon",
                        "in_context_privacy": False,
                        "performer_config_path": str(
                            Path(
                                code_dir,
                                "scripts/5-evaluation/config/ta2_performer_config.yaml",
                            )
                        ),
                        # these will be populated below.
                        "original_scores": "",
                        "original_scores_query_labels": "",
                        "original_scores_candidate_labels": "",
                        "privatized_scores": "",
                        "privatized_scores_query_labels": "",
                        "privatized_scores_candidate_labels": "",
                    }
                ],
            },
            # "soundness": {"metric_name": "perplexity", "model_id": "gpt2"},
            # "stylistic_consistency": None,
            # "stylistic_consistency": {
            #     "metric_name": "Harmonic Mean of Mean Percentile Rank",
            #     "ground_truth": paths["TA1"]["groundtruth"],
            #     "ta1_system_outputs": [
            #         {
            #             "ta1_system_name": "uoregon",
            #             "in_context_stylistic_consistency": False,
            #             # "performer_config_path": PERFORMER_CONFIG_PATH,
            #             # Add "privatized_features_dataset"
            #         }
            #     ],
            # },
        },
    }


####################################
# Main Evaluation
####################################
def evaluate_obfuscation(
    model,
    genre: str,
    language: str,
    output_dir: Path,
    batch_size: int = 32,
    pipeline_suffix: str = "prompt-remix",
    sampled_data_filepath: str = None,
    dataset: str = "hrs"
) -> Dict[str, Any]:
    """
    Runs TA3 evaluation for a single genre.
    - Extracts document-level embeddings for privatized queries and candidates
    - Extracts author-level embeddings (orig, privatized, candidates)
    - Computes TA2-style similarity scores for original and privatized author-level embeddings
    - Builds the TA3 config, populates paths (including stylistic_consistency), and runs the experiment
    """
    print(f"Running TA3 evaluation for {genre}")
    if dataset == "authbench" or "authbench" in pipeline_suffix.lower():
        ab_paths = authbench_paths(
            language=language,
            genre=genre,
            pipeline_suffix=pipeline_suffix,
        )
        priv_doc_path = ab_paths["privatized"]
        cand_doc_path = ab_paths["candidates"]
        queries_doc_path = ab_paths["queries"]
        gt_path = str(ab_paths["ground_truth"])
        gt_q_labels_path = None
        gt_c_labels_path = None
    else:
        paths = hrs_paths(
            genre=genre,
            language=language,
            pipeline_suffix=pipeline_suffix,
        )
        priv_doc_path = paths["TA3"]["data"]["privatized"]
        cand_doc_path = paths["TA3"]["data"]["candidates"]
        queries_doc_path = paths["TA3"]["data"]["queries"]
        gt_path = paths["TA3"]["groundtruth"]["groundtruth"]
        gt_q_labels_path = paths["TA3"]["groundtruth"]["query_labels"]
        gt_c_labels_path = paths["TA3"]["groundtruth"]["candidate_labels"]
    
    sampled_doc_ids = []

    if sampled_data_filepath is not None and os.path.exists(sampled_data_filepath):
        sample = pd.read_json(sampled_data_filepath)
        sampled_doc_ids = sample['documentID'].values.tolist()

    priv_doc = extract_embeddings(
        model,
        str(priv_doc_path),
        sampled_doc_ids,
        is_author_level=False,
        is_author_emb_model=True,
        batch_size=batch_size,
    )
    cand_doc = extract_embeddings(
        model,
        str(cand_doc_path),
        is_author_level=False,
        is_author_emb_model=True,
        batch_size=batch_size,
    )

    # Save document-level embeddings (these are the privatized features used by stylistic_consistency)
    doc_emb_path = output_dir / "doc_embeddings"
    DatasetDict({"queries": priv_doc, "candidates": cand_doc}).save_to_disk(
        str(doc_emb_path)
    )

    # --- Step 2: Author-level embeddings ---
    priv_auth = extract_embeddings(
        model,
        str(priv_doc_path),
        sampled_doc_ids,
        is_author_level=True,
        is_author_emb_model=True,
        batch_size=batch_size,
    )
    orig_auth = extract_embeddings(
        model,
        str(queries_doc_path),
        sampled_doc_ids,
        is_author_level=True,
        is_author_emb_model=True,
        batch_size=batch_size,
    )
    cand_auth = extract_embeddings(
        model,
        str(cand_doc_path),
        is_author_level=True,
        is_author_emb_model=True,
        batch_size=batch_size,
    )

    auth_emb_path = output_dir / "author_embeddings"
    DatasetDict(
        {"queries": orig_auth, "privatized_queries": priv_auth, "candidates": cand_auth}
    ).save_to_disk(str(auth_emb_path))

    # --- Step 3: Compute TA2 scores (original + privatized) ---
    orig_scores = compute_ta2_scores(
        auth_emb_path, output_dir / "ta3_results_orig", query_split="queries"
    )
    priv_scores = compute_ta2_scores(
        auth_emb_path, output_dir / "ta3_results_priv", query_split="privatized_queries"
    )

    # --- Step 4: Build config and run experiment ---
    paths_dict = None if dataset == "authbench" or "authbench" in pipeline_suffix.lower() else paths
    if gt_q_labels_path is None:
        gt_q_labels_path = orig_scores["query_labels"]
    if gt_c_labels_path is None:
        gt_c_labels_path = orig_scores["candidate_labels"]
        
    if dataset == "authbench" or "authbench" in pipeline_suffix.lower():
        with open(gt_q_labels_path) as f:
            q_lbls = [line.strip() for line in f]
        with open(gt_c_labels_path) as f:
            c_lbls = [line.strip() for line in f]
            
        auth_gt = np.zeros((len(q_lbls), len(c_lbls)), dtype=int)
        for i, q in enumerate(q_lbls):
            for j, c in enumerate(c_lbls):
                if q == c:
                    auth_gt[i, j] = 1
        
        gt_path = str(output_dir / "auth_gt_matrix.npy")
        np.save(gt_path, auth_gt)
    
    config = generate_evaluation_config(paths_dict, pipeline_suffix, gt_path, gt_q_labels_path, gt_c_labels_path)
    ta2_config = config["ta3"]["privacy"]["ta2_system_outputs"][0]

    ta2_config["original_scores"] = orig_scores["scores"]
    ta2_config["original_scores_query_labels"] = orig_scores["query_labels"]
    ta2_config["original_scores_candidate_labels"] = orig_scores["candidate_labels"]

    ta2_config["privatized_scores"] = priv_scores["scores"]
    ta2_config["privatized_scores_query_labels"] = priv_scores["query_labels"]
    ta2_config["privatized_scores_candidate_labels"] = priv_scores["candidate_labels"]

    if "stylistic_consistency" in config["ta3"]:
        config["ta3"]["stylistic_consistency"]["ta1_system_outputs"][0][
            "privatized_features_dataset"
        ] = (Path(doc_emb_path).expanduser().as_posix())

    if "soundness" in config["ta3"]:
        try:
            config["ta3"]["soundness"]["original_dataset"] = str(
                paths["TA3"]["data"]["queries"]
            )
            config["ta3"]["soundness"]["privatized_dataset"] = str(
                paths["TA3"]["data"]["privatized"]
            )
        except Exception:
            pass

    # --- Step 5: Run TA3 experiment ---
    experiment = Experiment(config=config)
    _, ta3_results, _ = experiment.compute_metrics()
    print(f"TA3 results for {genre}:")
    print(ta3_results)

    return ta3_results


####################################
# CLI
####################################
def main(args: argparse.Namespace) -> None:
    """CLI entrypoint: load model, run evaluation for a genre, and save results."""

    model = transformers.AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.cuda()
    model.eval()

    result = evaluate_obfuscation(
        model=model,
        genre=args.genre,
        language=args.language,
        pipeline_suffix=args.pipeline_suffix,  # now explicit
        output_dir=metrics_dir,
        batch_size=args.batch_size,
        sampled_data_filepath=args.sampled_data_filepath,
        dataset=args.dataset
    )

    pd.DataFrame([result]).to_csv(
        metrics_dir / args.language /f"{args.genre}_{args.pipeline_suffix}_ta3_results.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--genre", type=str, required=True)
    parser.add_argument(
        "--language", type=str, required=True, choices=["en", "ru", "zh", "ar"]
    )
    parser.add_argument(
        "--pipeline-suffix",
        type=str,
        default="",
        help="Pipeline suffix for this evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampled-data-filepath", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="hrs", choices=["hrs", "authbench"])

    args = parser.parse_args()

    main(args)
