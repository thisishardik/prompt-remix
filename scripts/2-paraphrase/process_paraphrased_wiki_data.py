import argparse
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import pycld2 as cld2
from tqdm import tqdm

from hiatus.utils import df_to_jsonl, load_data

warnings.filterwarnings(
    "ignore",
    "This pattern is interpreted as a regular expression, and has match groups",
)


def select_columns_to_filter(prompts: pd.DataFrame) -> List:
    """Return column indices that contain generated paraphrase text."""
    if len(prompts.columns) == 4:
        columns_to_filter = [2, 3]  # axis_more/axis_les
    elif len(prompts.columns) == 3:
        columns_to_filter = [2]  # axis_more
    else:
        raise ValueError
    return columns_to_filter


def remove_reasoning_marks(
    prompts: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Remove reasoning markers from paraphrase text."""
    columns_to_filter = select_columns_to_filter(prompts)

    # Remove normal reasoning markers
    prompts.iloc[:, columns_to_filter] = prompts.iloc[:, columns_to_filter].map(
        lambda text: text.split("</think>\n\n")[-1]
    )

    # Remove normal reasoning markers
    has_abnormal_reasoning = (
        prompts.iloc[:, columns_to_filter]
        .astype(str)
        .apply(lambda row: row.str.contains("</think>|<\/think>|<think>").any(), axis=1)
    )
    if verbose:
        print("\t abnormal reasoning marks:", (has_abnormal_reasoning).sum())
    return prompts[~has_abnormal_reasoning]


def is_valid_language(text: str, lang: str, lang_threshold: float = 0.7) -> bool:
    """Return True if `text` is detected as language `lang` with confidence."""
    try:
        is_reliable, text_bytes, top_languages = cld2.detect(text, bestEffort=False)
        languages = {
            language_code: percent
            for (language_name, language_code, percent, score) in top_languages
        }
        if lang == "zh":
            language_present = is_reliable and (
                "zh" in languages or "zh-Hant" in languages
            )
            language_confident = (
                languages.get("zh", 0) + languages.get("zh-Hant", 0)
            ) / 100.0 > lang_threshold
        else:
            language_present = is_reliable and (lang in languages)
            language_confident = languages.get(lang, 0) / 100.0 > lang_threshold
        return language_present and language_confident
    except Exception:
        return False


def filter_repeating_characters(
    prompts: pd.DataFrame, max_repeat: int = 10, verbose: bool = False
) -> pd.DataFrame:
    """Remove rows where paraphrases contain long runs of repeated characters."""
    columns_to_filter = select_columns_to_filter(prompts)
    has_repeating_characters = (
        prompts.iloc[:, columns_to_filter]
        .astype(str)
        .apply(
            lambda row: row.str.contains(
                r"(.)\1{%d,}" % (max_repeat - 1), regex=True
            ).any(),
            axis=1,
        )
    )
    if verbose:
        print("\t repeating characters:", has_repeating_characters.sum())
    return prompts[~has_repeating_characters]


def filter_language(prompts: pd.DataFrame, lang: str, verbose: bool = False) -> pd.DataFrame:
    """Keep only rows where all target columns pass language detection."""
    columns_to_filter = select_columns_to_filter(prompts)
    has_valid_language = (
        prompts.iloc[:, columns_to_filter]
        .map(lambda text: is_valid_language(text, lang))
        .all(axis=1)
    )
    if verbose:
        print("\t invalid language:", (~has_valid_language).sum())
    return prompts[has_valid_language]


def main(args: argparse.Namespace) -> None:
    """Process all paraphrased jsonl files in a folder and save cleaned outputs."""
    paraphrased_paths = list(Path(args.input_dir).glob("*.jsonl"))
    for path in tqdm(paraphrased_paths):
        print("style axis:", str(path.name).split("_")[0])
        paraphrases_df = load_data(str(path))
        processed_paraphrases = (
            paraphrases_df.pipe(remove_reasoning_marks, verbose=args.verbose)
            .pipe(filter_language, lang=args.lang, verbose=args.verbose)
            .pipe(filter_repeating_characters, verbose=args.verbose)
        )
        output_path = Path(args.output_dir, path.name)
        df_to_jsonl(processed_paraphrases, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        "-i",
        default="/gscratch/stf/mpotto/hiatus/data/zh/paraphrases/original",
        help="Path to folder with paraphrased prompts.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="/gscratch/stf/mpotto/hiatus/data/zh/paraphrases/processed",
        help="Path to folder with processed paraphrased prompts.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="zh"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print how many entries were removed in the process.",
    )
    args = parser.parse_args()
    main(args)
