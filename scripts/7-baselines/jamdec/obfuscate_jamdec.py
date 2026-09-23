import argparse
import json
from pathlib import Path
from tqdm import tqdm

from hiatus.data import authbench_paths, blog_authorship_paths, hrs_paths
from jamdec_method import JAMDECGeneration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["authbench", "blog", "hrs"])
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--sample-name", type=str, default=None)
    parser.add_argument("--output-suffix", type=str, default="jamdec")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--sampled-data-filepath", type=str, default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced JAMDEC search (much faster; for large AuthBench runs).",
    )
    
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
            sampled_data = json.load(f)
            sampled_doc_ids = {item["documentID"] for item in sampled_data}
        queries = [q for q in queries if q.get("documentID") in sampled_doc_ids]
        
    if args.limit:
        queries = queries[:args.limit]

    output_path = paths["privatized"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = row.get("documentID")
                if doc_id:
                    done_ids.add(doc_id)
        print(f"Resuming {output_path}: {len(done_ids)} queries already done")
        queries = [q for q in queries if q.get("documentID") not in done_ids]

    print("Initializing JAMDEC Generator...")
    jamdec_generator = JAMDECGeneration(args=args)

    print(f"Running JAMDEC Obfuscation on {len(queries)} remaining queries...")
    with open(output_path, "a") as f:
        for q in tqdm(queries):
            query_text = q["fullText"]
            try:
                gen, candidate_data, gen_cola = jamdec_generator.generate(prompt=query_text)
                obfuscated_text = gen[0]
            except Exception as e:
                print(f"Error obfuscating query {q.get('documentID', 'unknown')}: {e}")
                obfuscated_text = query_text

            q_copy = q.copy()
            q_copy["fullText"] = obfuscated_text
            f.write(json.dumps(q_copy) + "\n")
            f.flush()

    print(f"Privatized queries saved to {output_path}")

if __name__ == "__main__":
    main()
