import pandas as pd
import argparse
import random
import pprint

def process_data(input_path, output_path, input_key, seed, output_key=None):
    # Set the random seed if one was provided
    if seed is not None:
        random.seed(seed)
        print(f"Random seed set to: {seed}")

    # 1. Read JSONL data
    print(f"Reading data from {input_path}...")
    df = pd.read_json(input_path, lines=True)

    # 2. Select a random authorID
    # Explode the lists to get a flat array of all unique authors
    all_authors = df['authorIDs'].explode().dropna().unique()
    
    if len(all_authors) == 0:
        print("Error: No author IDs found in the dataset.")
        return

    random_author = random.choice(all_authors)
    print(f"Selected Author ID: {random_author}")

    # 3. Take all records for that author
    # Filter rows where the random_author is present in the authorIDs list
    author_df = df[df['authorIDs'].apply(lambda x: isinstance(x, list) and random_author in x)].copy()
    print(f"Found {len(author_df)} records for this author.")

    output_df = author_df[[input_key]].copy()
    if output_key is not None:
        output_df = output_df.rename(columns={input_key: output_key})
        
    output_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Successfully saved records to {output_path}")

def main():
    # 5. Take parameters for input path, output path, input key, and seed
    parser = argparse.ArgumentParser(description="Extract text records for a random author from a JSONL file.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL dataset.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the output JSONL.")
    parser.add_argument("--input-key", type=str, required=True, help="The JSON key containing the target text (e.g., 'fullText').")
    parser.add_argument("--output-key", type=str, default=None, help="The JSON key to use for the output text (default: 'fullText').")
    parser.add_argument("--seed", type=int, default=42, help="Optional integer for random seed to ensure reproducibility.")

    args = parser.parse_args()
    pprint.pprint(vars(args))

    process_data(args.input_path, args.output_path, args.input_key, args.seed, args.output_key)

if __name__ == "__main__":
    main()