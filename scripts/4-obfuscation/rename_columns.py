from argparse import ArgumentParser

from datasets import load_dataset

def main() -> None:
    """CLI entrypoint: rename a column in a jsonl dataset and save result."""
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--old-key", type=str, required=True)
    parser.add_argument("--new-key", type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.input_path)["train"]
    if args.new_key in dataset.column_names:
        dataset = dataset.remove_columns(args.new_key)
    dataset = dataset.rename_column(args.old_key, args.new_key)

    dataset.to_json(args.output_path, orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":
    main()
