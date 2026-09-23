import os
import json
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
from functools import partial

from hiatus.data import authbench_paths, blog_authorship_paths, hrs_paths
from hiatus.obfuscation.evaluation import EvaluationRunner
from hiatus.obfuscation.directions import choose_sliders_styleremix
from styleremix_method import StyleRemixGeneration
import sys

STYLE_REMIX_ROOT = "/mmfs1/home/hardiksr/repository/StyleRemix"
if STYLE_REMIX_ROOT not in sys.path:
    sys.path.insert(0, STYLE_REMIX_ROOT)
from quickstart import MODEL_PATHS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["authbench", "blog", "hrs"])
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--sample-name", type=str, default=None)
    parser.add_argument("--output-suffix", type=str, default="styleremix-baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sampled-data-filepath", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--top-n-styles-to-change", type=int, default=3)
    parser.add_argument("--model-ckpt", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--config-file",
        type=str,
        default="/mmfs1/home/hardiksr/repository/hiatus/scripts/4-obfuscation/config/remix_config_en.json",
    )
    
    args = parser.parse_args()

    if args.dataset == "authbench":
        paths = authbench_paths(language=args.language, genre=args.genre, pipeline_suffix=args.output_suffix)
    elif args.dataset == "blog":
        paths = blog_authorship_paths(sample_name=args.sample_name, pipeline_suffix=args.output_suffix)
    elif args.dataset == "hrs":
        paths = hrs_paths(genre=args.genre, language=args.language, pipeline_suffix=args.output_suffix)["TA3"]["data"]

    print(f"Loading queries from {paths['queries']}")
    with open(paths["queries"], "r") as f:
        queries = [json.loads(line) for line in f]
        
    if args.sampled_data_filepath:
        with open(args.sampled_data_filepath, "r") as f:
            sampled_docs = json.load(f)
        sampled_doc_ids = {doc["documentID"] for doc in sampled_docs}
        queries = [q for q in queries if q.get("documentID") in sampled_doc_ids]
        
    if args.limit:
        queries = queries[:args.limit]

    data = pd.DataFrame(queries)
    if "authorIDs" not in data.columns:
        raise ValueError("Data must contain 'authorIDs' for per-author StyleRemix evaluation.")
        
    author_data = data.explode("authorIDs", ignore_index=True)
    author_texts = author_data.groupby("authorIDs")["fullText"].apply(list)
    
    print("Running automatic evaluation to calculate per-author weights...")
    with open(args.config_file, "r") as f:
        styleremix_config = json.load(f)
        
    evaluation_runner = EvaluationRunner(
        classifier_config=styleremix_config["classifiers"],
        lang=args.language,
        library="stanza" if args.language == "ar" else "spacy",
    )
    
    all_author_scores = {}
    for author_id, texts in tqdm(author_texts.items(), desc="Evaluating Authors"):
        author_scores = evaluation_runner(texts)
        all_author_scores[author_id] = author_scores
    evaluation_runner.cleanup()
    
    author_scores = pd.DataFrame.from_dict(all_author_scores, orient="index")
    normalized_scores = (author_scores / author_scores.max(axis=0)).fillna(0)
    score_std = normalized_scores.std(axis=0)
    
    if args.language in ["en", "ru"]:
        types = ["type_persuasive", "type_narrative", "type_expository", "type_descriptive"]
    else:
        types = ["persuasive_more", "narrative_more", "expository_more", "descriptive_more"]
        
    available_types = [type_name for type_name in types if type_name in score_std and score_std[type_name] > 0]
    
    author_directions = choose_sliders_styleremix(
        author_scores,
        normalized_scores,
        args.top_n_styles_to_change,
        score_std
    )

    styleremix_generator = StyleRemixGeneration(args=args)

    privatized_queries = []
    print(f"Running StyleRemix Obfuscation on {len(queries)} queries...")

    for q in tqdm(queries):
        query_text = q["fullText"]
        author_id = q["authorIDs"][0] if isinstance(q["authorIDs"], list) else q["authorIDs"]
        author_weights = author_directions.get(author_id, {})
        filtered_weights = {}
        for k, v in author_weights.items():
            adapter_key = k
            if v < 0:
                adapter_key = k.replace("_more", "_less")
            if adapter_key in MODEL_PATHS:
                filtered_weights[adapter_key] = abs(v)
            elif k in MODEL_PATHS:
                filtered_weights[k] = v

        try:
            gen, metadata, _ = styleremix_generator.generate(prompt=query_text, author_weights=filtered_weights)
            obfuscated_text = gen[0]
        except Exception as e:
            print(f"Error obfuscating query {q.get('documentID', 'unknown')}: {e}")
            obfuscated_text = query_text
            
        q_copy = q.copy()
        q_copy["fullText"] = obfuscated_text
        q_copy["StyleRemix_weights"] = author_weights
        privatized_queries.append(q_copy)

    output_path = paths["privatized"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for q in privatized_queries:
            f.write(json.dumps(q) + "\n")
            
    print(f"Privatized queries saved to {output_path}")

if __name__ == "__main__":
    main()
