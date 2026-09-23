import argparse
import json
from pathlib import Path
from tqdm import tqdm

from hiatus.data import authbench_paths, blog_authorship_paths, hrs_paths
from steer_method import STEERGeneration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["authbench", "blog", "hrs"])
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--sample-name", type=str, default=None)
    parser.add_argument("--output-suffix", type=str, default="steer")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sampled-data-filepath", type=str, default=None)
    parser.add_argument("--target-style", type=str, default="english_tweet",
                        choices=["english_tweet", "romantic_poetry", "shakespeare", "aae", 
                                 "bible", "coha_1810", "coha_1890", "coha_1990", "joyce", 
                                 "lyrics", "switchboard"])
    parser.add_argument("--model-ckpt", type=str, default="/gscratch/stf/kogolobo/ckp_3500.pth")
    parser.add_argument("--base-model", type=str, default="gpt2-large")
    parser.add_argument("--max-prompt-length", type=int, default=50)
    parser.add_argument("--max-gen-length", type=int, default=100)
    parser.add_argument("--no-repeat-ngrams", type=int, default=3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--n-extra-tokens", type=int, default=5)
    
    args = parser.parse_args()

    if args.dataset == "authbench":
        paths = authbench_paths(language=args.language, genre=args.genre, pipeline_suffix=args.output_suffix)
    elif args.dataset == "blog":
        paths = blog_authorship_paths(sample_name=args.sample_name, pipeline_suffix=args.output_suffix)
    elif args.dataset == "hrs":
        paths = hrs_paths(genre=args.genre, language=args.language, pipeline_suffix=args.output_suffix)["TA3"]["data"]

    with open(paths["queries"], "r") as f:
        queries = [json.loads(line) for line in f]
        
    if args.sampled_data_filepath:
        with open(args.sampled_data_filepath, "r") as f:
            sampled_docs = json.load(f)
        sampled_doc_ids = {doc["documentID"] for doc in sampled_docs}
        queries = [q for q in queries if q.get("documentID") in sampled_doc_ids]
        
    if args.limit:
        queries = queries[:args.limit]

    print(f"Initializing STEER Generator (target={args.target_style})...")
    steer_generator = STEERGeneration(args=args)

    privatized_queries = []

    print(f"Running STEER Obfuscation on {len(queries)} queries...")
    for q in tqdm(queries):
        query_text = q["fullText"]
        try:
            gen, candidate_data, gen_cola = steer_generator.generate(prompt=query_text)
            obfuscated_text = gen[0]
        except Exception as e:
            print(f"Error obfuscating query {q.get('documentID', 'unknown')}: {e}")
            obfuscated_text = query_text
            
        q_copy = q.copy()
        q_copy["fullText"] = obfuscated_text
        privatized_queries.append(q_copy)

    output_path = paths["privatized"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for q in privatized_queries:
            f.write(json.dumps(q) + "\n")
            
    print(f"Privatized queries saved to {output_path}")

if __name__ == "__main__":
    main()
