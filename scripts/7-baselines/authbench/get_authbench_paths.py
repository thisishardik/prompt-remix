from argparse import ArgumentParser
from pathlib import Path

from hiatus.data import authbench_paths


def main():
    parser = ArgumentParser(description="Print AuthBench file paths")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["en", "ru", "zh", "ar"],
    )
    parser.add_argument(
        "--genre",
        type=str,
        default=None,
        help="Genre filter (e.g. 'social_media/technology'). If omitted, uses full language-level data.",
    )
    parser.add_argument(
        "--pipeline-suffix",
        type=str,
        default=None,
        help=(
            "Pipeline suffix for privatized output file naming. Defaults to "
            "promptremix_authbench-{language}0-3."
        ),
    )
    args = parser.parse_args()

    paths = authbench_paths(
        language=args.language,
        genre=args.genre,
        pipeline_suffix=args.pipeline_suffix,
    )

    print(
        paths["queries"],
        paths["privatized"],
        paths["candidates"],
        paths["ground_truth"],
    )


if __name__ == "__main__":
    main()
