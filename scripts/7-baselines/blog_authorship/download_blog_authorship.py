import os
import json
import random
import zipfile
import urllib.request
import numpy as np
from pathlib import Path
import glob

def parse_blog_file(file_path):
    file_name = os.path.basename(file_path)
    parts = file_name.split(".")[:-1]
    if len(parts) != 5:
        return []
    
    file_id, gender, age, job, horoscope = parts
    posts = []
    
    with open(file_path, encoding="latin_1") as f:
        current_date = ""
        for line in f:
            line = line.strip()
            if "<date>" in line:
                current_date = line.replace("<date>", "").replace("</date>", "").strip()
            elif line != "" and not line.startswith("<"):
                posts.append({
                    "text": line,
                    "id": file_id,
                    "gender": gender,
                    "age": int(age),
                    "job": job,
                    "horoscope": horoscope,
                    "date": current_date
                })
    return posts

def prepare_blog_authorship(output_dir, num_authors=100, queries_per_author=1, candidates_per_author=10):
    output_path = Path(output_dir)
    ta3_dir = output_path / "TA3"
    data_dir = ta3_dir / "data"
    gt_dir = ta3_dir / "groundtruth"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    zip_url = "https://huggingface.co/datasets/barilan/blog_authorship_corpus/resolve/main/data/blogs.zip"
    zip_path = output_path / "blogs.zip"
    extract_dir = output_path / "blogs_extracted"

    if not zip_path.exists():
        print(f"Downloading dataset from {zip_url}...")
        urllib.request.urlretrieve(zip_url, zip_path)
    
    if not extract_dir.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    print("Grouping texts by author...")
    author_map = {}
    xml_files = glob.glob(str(extract_dir / "blogs" / "*.xml"))
    
    for file_path in xml_files:
        posts = parse_blog_file(file_path)
        if not posts:
            continue
        
        author_id = posts[0]['id']
        if author_id not in author_map:
            author_map[author_id] = []
        
        for post in posts:
            author_map[author_id].append(post['text'])

    authors = list(author_map.keys())
    print(f"Total unique authors found: {len(authors)}")
    
    min_texts = queries_per_author + candidates_per_author
    valid_authors = [a for a in authors if len(author_map[a]) >= min_texts]
    print(f"Authors with at least {min_texts} texts: {len(valid_authors)}")
    
    if len(valid_authors) < num_authors:
        print(f"Warning: Requested {num_authors} authors, but only {len(valid_authors)} are valid.")
        num_authors = len(valid_authors)
    
    sampled_authors = random.sample(valid_authors, num_authors)
    
    queries = []
    candidates = []
    
    print(f"Sampling {num_authors} authors and generating query/candidate sets...")
    for author_id in sampled_authors:
        texts = author_map[author_id]
        random.shuffle(texts)
        
        author_queries = texts[:queries_per_author]
        author_candidates = texts[queries_per_author:queries_per_author + candidates_per_author]
        
        for i, text in enumerate(author_queries):
            doc_id = f"blog_{author_id}_q{i}"
            queries.append({
                "documentID": doc_id,
                "authorIDs": [author_id],
                "fullText": text,
                "lang": "en"
            })
            
        for i, text in enumerate(author_candidates):
            doc_id = f"blog_{author_id}_c{i}"
            candidates.append({
                "documentID": doc_id,
                "authorIDs": [author_id],
                "fullText": text,
                "lang": "en"
            })
            
    print("Saving queries and candidates to JSONL...")
    with open(data_dir / "input_queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
            
    with open(data_dir / "input_candidates.jsonl", "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    print("Generating ground truth matrix and label files...")
    num_queries = len(queries)
    num_candidates = len(candidates)
    gt_matrix = np.zeros((num_queries, num_candidates), dtype=int)
    
    query_labels = []
    for i, q in enumerate(queries):
        query_author = q["authorIDs"][0]
        query_labels.append(tuple([query_author]))
        for j, c in enumerate(candidates):
            candidate_author = c["authorIDs"][0]
            if query_author == candidate_author:
                gt_matrix[i, j] = 1
                
    candidate_labels = []
    for c in candidates:
        candidate_labels.append(tuple([c["authorIDs"][0]]))
        
    np.save(gt_dir / "groundtruth.npy", gt_matrix)
    
    with open(gt_dir / "query-labels.txt", "w") as f:
        for label in query_labels:
            f.write(f"{label}\n")
            
    with open(gt_dir / "candidate-labels.txt", "w") as f:
        for label in candidate_labels:
            f.write(f"{label}\n")
            
    print("Successfully prepared Blog Authorship Corpus sample.")
    print(f"Queries: {num_queries}, Candidates: {num_candidates}")
    print(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_authors", type=int, default=100)
    parser.add_argument("--queries_per_author", type=int, default=1)
    parser.add_argument("--candidates_per_author", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    prepare_blog_authorship(
        args.output_dir, 
        args.num_authors, 
        args.queries_per_author, 
        args.candidates_per_author
    )
