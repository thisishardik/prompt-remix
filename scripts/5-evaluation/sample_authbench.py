import json
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import defaultdict

from hiatus.data import authbench_paths

def load_jsonl(filepath: str) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def main(language: str, genre: str, output_dir: str, num_authors: int = 50, max_docs_per_author: int = 2, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    paths = authbench_paths(language=language, genre=genre, pipeline_suffix="dummy")
    queries_path = paths["queries"]
    
    print(f"Loading data for {language} - {genre}...")
    
    if not Path(queries_path).exists():
        print(f"Skipping (Missing original queries): {queries_path}")
        return
        
    queries = load_jsonl(str(queries_path))
    
    author_to_indices = defaultdict(list)
    for i, q in enumerate(queries):
        author_key = next((k for k in ["authorSetIDs", "authorIDs", "authorID", "author"] if k in q), None)
        if not author_key:
            raise KeyError(f"Could not find author key in query: {q.keys()}")
            
        author_val = q[author_key]
        author_label = author_val[0] if isinstance(author_val, list) else author_val
        author_to_indices[author_label].append(i)
        
    eligible_authors = [author for author, indices in author_to_indices.items() if len(indices) >= 1]
    print(f"Found {len(eligible_authors)} authors with at least 1 document.")
    
    if len(eligible_authors) == 0:
        print(f"Skipping (No eligible authors)")
        return
        
    num_to_sample = min(num_authors, len(eligible_authors))
    if num_to_sample < num_authors:
        print(f"WARNING: Only {num_to_sample} eligible authors found. Taking as many as possible.")
        
    sampled_authors = random.sample(eligible_authors, num_to_sample)
    
    sampled_query_indices = []
    for author in sampled_authors:
        author_indices = author_to_indices[author]
        num_docs_to_sample = random.randint(1, min(len(author_indices), max_docs_per_author))
        sampled_docs = random.sample(author_indices, num_docs_to_sample)
        sampled_query_indices.extend(sampled_docs)
        
    sampled_query_indices.sort()
    
    print(f"Sampled {len(sampled_authors)} authors, total {len(sampled_query_indices)} documents.")
    
    sampled_docs_info = []
    for i in sampled_query_indices:
        q = queries[i]
        doc_id_key = next((k for k in ["documentID", "document_id", "id"] if k in q), None)
        doc_id = q[doc_id_key] if doc_id_key else None
        
        sampled_docs_info.append({
            "index": i,
            "documentID": doc_id,
        })
    
    out_dir = Path(output_dir) / language / genre
    out_dir.mkdir(parents=True, exist_ok=True)
    
    genre_safe = genre.replace("/", "_")
    out_path = out_dir / f"authbench_{language}_{genre_safe}_sampled_ids.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sampled_docs_info, f, indent=2)
        
    print(f"Saved sampled document ids to {out_path}")

if __name__ == "__main__":
    LANGUAGE_GENRES = {
        "en": ["blog", "ecommerce_reviews", "literature", "news", "poetry", "qna", "research_paper"],
        "ar": ["literature", "news", "poetry", "social_media"],
        "zh": ["ecommerce_reviews", "literature", "media_reviews", "news", "poetry", "social_media"],
        "ru": ["literature", "news", "poetry", "qna", "social_media"]
    }
    
    BASE_OUTPUT_DIR = "/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/"
    
    for language, genres in LANGUAGE_GENRES.items():
        for genre in genres:
            print(f"Processing {language.upper()} - {genre}")
            
            main(
                language=language,
                genre=genre,
                output_dir=BASE_OUTPUT_DIR,
                num_authors=50,
                max_docs_per_author=2,
                seed=42
            )
