from argparse import ArgumentParser
from typing import Literal, Optional

from hiatus.data import hrs_paths


def get_input_privatized_paths(
    genre: Literal[
        "AMINA",
        "bbn_aljazeera_blog_1",
        "bbn_ida2at",
        "kaggle_arabic_news_1",
        "ar_twitter_1",
        "saudinewsnet",
        "ru_crossGenre",
        "en_crossGenre",
        "en_short_crossGenre",
        "zh_short_crossGenre",
        "zh_medium_crossGenre",
        "perGenre-HRS2.1",
        "perGenre-HRS2.2",
        "perGenre-HRS2.3",
        "perGenre-HRS2.4",
        "perGenre-HRS2.5",
        "perGenre-HRS3.1",
        "perGenre-HRS3.2",
        "perGenre-HRS3.3",
        "perGenre-HRS3.4",
        "perGenre-HRS3.5",
        "perGenre-HRS2.101",
        "perGenre-HRS2.102",
        "perGenre-HRS2.103",
        "perGenre-HRS2.104",
        "perGenre-HRS2.105",
        "perGenre-HRS3.301",
        "perGenre-HRS3.302",
        "perGenre-HRS3.303",
        "perGenre-HRS3.304",
        "perGenre-HRS3.305",
    ],
    language: Literal["en", "ru", "zh", "ar"],
    pipeline_suffix: Optional[str] = "prompt-remix",
) -> None:
    """Print the input and privatized query paths for a given HRS config."""
    paths = hrs_paths(
        genre=genre,
        language=language,
        pipeline_suffix=pipeline_suffix,
    )
    ta3_data = paths["TA3"]["data"]
    input_queries_path, privatized_queries_path, candidate_queries_path = (
        ta3_data["queries"],
        ta3_data["privatized"],
        ta3_data["candidates"],
    )
    ground_truth_path = paths['TA3']['groundtruth']['groundtruth']
    print(input_queries_path, privatized_queries_path, candidate_queries_path, ground_truth_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--genre",
        type=str,
        choices=[
            "AMINA",
            "bbn_aljazeera_blog_1",
            "bbn_ida2at",
            "kaggle_arabic_news_1",
            "ar_twitter_1",
            "saudinewsnet",
            "ru_crossGenre",
            "en_crossGenre",
            "en_short_crossGenre",
            "zh_short_crossGenre",
            "zh_medium_crossGenre",
            "perGenre-HRS2.1",
            "perGenre-HRS2.2",
            "perGenre-HRS2.3",
            "perGenre-HRS2.4",
            "perGenre-HRS2.5",
            "perGenre-HRS3.1",
            "perGenre-HRS3.2",
            "perGenre-HRS3.3",
            "perGenre-HRS3.4",
            "perGenre-HRS3.5",
            "perGenre-HRS2.101",
            "perGenre-HRS2.102",
            "perGenre-HRS2.103",
            "perGenre-HRS2.104",
            "perGenre-HRS2.105",
            "perGenre-HRS3.301",
            "perGenre-HRS3.302",
            "perGenre-HRS3.303",
            "perGenre-HRS3.304",
            "perGenre-HRS3.305",
        ],
        required=True,
    )
    parser.add_argument("--language", type=str, choices=["en", "ru", "zh", "ar"])
    parser.add_argument("--pipeline-suffix", type=str)
    args = parser.parse_args()

    get_input_privatized_paths(
        genre=args.genre, language=args.language, pipeline_suffix=args.pipeline_suffix
    )
