from argparse import ArgumentParser
from hiatus.data import blog_authorship_paths

def main():
    parser = ArgumentParser(description="Print Blog Authorship file paths")
    parser.add_argument(
        "--sample-name",
        type=str,
        default="sample_100",
        help="Name of the processed sample (e.g. 'sample_100').",
    )
    parser.add_argument(
        "--pipeline-suffix",
        type=str,
        default="promptremix-blog",
        help="Pipeline suffix for privatized output file naming.",
    )
    args = parser.parse_args()

    paths = blog_authorship_paths(
        sample_name=args.sample_name,
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
