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

    if 'authorIDs' not in df.columns or input_key not in df.columns:
        print(f"Error: Required columns ('authorIDs' and '{input_key}') not found in the dataset.")
        return

    # 2. Explode the lists so each author-document pair has its own row
    df_exploded = df.explode('authorIDs')
    
    # Drop rows where authorID is missing or the target text is empty
    df_exploded = df_exploded.dropna(subset=['authorIDs', input_key])
    
    unique_authors_count = df_exploded['authorIDs'].nunique()
    print(f"Found {unique_authors_count} unique authors.")

    # 3. For each author, sample 1 text randomly
    print("Sampling 1 random text per author...")
    sampled_df = df_exploded.groupby('authorIDs').sample(n=1, random_state=seed)

    # 4. Prepare and save the output
    output_df = sampled_df[[input_key]].copy()
    if output_key is not None:
        output_df = output_df.rename(columns={input_key: output_key})
        
    output_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Successfully saved {len(output_df)} genre examples to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract 1 random text per author from a JSONL file to create a genre examples file.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL dataset.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the output JSONL.")
    parser.add_argument("--input-key", type=str, required=True, help="The JSON key containing the target text (e.g., 'fullText').")
    parser.add_argument("--output-key", type=str, default=None, help="The JSON key to use for the output text.")
    parser.add_argument("--seed", type=int, default=42, help="Optional integer for random seed to ensure reproducibility.")

    args = parser.parse_args()
    pprint.pprint(vars(args))

    process_data(args.input_path, args.output_path, args.input_key, args.seed, args.output_key)

if __name__ == "__main__":
    main()