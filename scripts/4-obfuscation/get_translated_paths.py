from argparse import ArgumentParser
from pathlib import Path

from hiatus.data import translated_paths


def main(args):
    input_path = Path(args.input_path)
    filepath = translated_paths(
        input_path=input_path,
        source=args.source,
        target=args.target,
    )
    print(filepath)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--source", type=str, default="en")
    parser.add_argument("--target", type=str, default="ar")
    args = parser.parse_args()
    main(args)