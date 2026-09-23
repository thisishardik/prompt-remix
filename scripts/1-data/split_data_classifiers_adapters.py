import argparse

import json
from pathlib import Path

import pandas as pd


def df_to_jsonl(df: pd.DataFrame, fname: str) -> None:
    """
    Write a DataFrame to a JSONL file.
    """
    output_path = Path(fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, index=False, lines=True, orient="records", force_ascii=False)

def main(args: argparse.Namespace) -> None:
    """
    Load JSON records, shuffle deterministically, split in half, and save two jsonl files.
    """
    filepath = Path(args.input_path)
    with open(filepath, "r") as file:
        full_data = json.load(file)

    # Shuffle deterministically and split based on n_samples
    full_data = pd.DataFrame(full_data).sample(frac=1, random_state=0)

    adapter = pd.DataFrame(full_data[: args.n_samples])
    classifier = pd.DataFrame(full_data[args.n_samples: 2*args.n_samples])

    df_to_jsonl(adapter, args.adapter_base_texts)
    df_to_jsonl(classifier, args.classifier_base_texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples in either the classifier or the adapter split."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/gscratch/stf/mpotto/hiatus/data/zh/raw/wiki.jsonl",
    )
    parser.add_argument(
        "--adapter-base-texts",
        type=str,
        default="/gscratch/stf/mpotto/hiatus/data/zh/raw/wiki_adapters.jsonl",
    )
    parser.add_argument(
        "--classifier-base-texts",
        type=str,
        default="/gscratch/stf/mpotto/hiatus/data/zh/raw/wiki_classifiers.jsonl",
    )
    args = parser.parse_args()
    main(args)
