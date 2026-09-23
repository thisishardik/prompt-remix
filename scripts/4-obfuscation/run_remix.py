import os
import json
import pprint
import random
from argparse import ArgumentParser
from functools import partial

import numpy as np
import pandas as pd
import tomli
from tqdm.auto import tqdm

from hiatus.obfuscation.evaluation import EvaluationRunner
from hiatus.obfuscation.directions import (
    choose_random_directions,
    combine_directions,
    choose_sliders_round_robin,
    choose_directions_target_author,
)
from hiatus.obfuscation.style_remix import RemixRunner
from hiatus.obfuscation.prompt_remix import PromptRemixRunner


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, default="data.jsonl")
    parser.add_argument("--model-id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument(
        "--lang", type=str, choices=["en", "ru", "zh", "ar"], default="en"
    )
    parser.add_argument(
        "--library", type=str, choices=["spacy", "stanza", "corenlp"], default="spacy"
    )
    parser.add_argument("--prompt-remix", action="store_true")
    parser.add_argument(
        "--config-file",
        type=str,
        default="/mmfs1/home/mpotto/code/hiatus/scripts/4-obfuscation/config/remix_config_zh.json",
    )
    parser.add_argument("--text-key", type=str, default="fullText")
    parser.add_argument("--author-key", type=str, default="authorIDs")
    parser.add_argument("--document-key", type=str, default="documentID")
    parser.add_argument("--output-key", type=str, default="remixedText")
    parser.add_argument("--output-file", type=str, default="output.jsonl")

    # Genre input argument
    parser.add_argument("--genre-input-file", type=str, default=None, help="Path to jsonl file with target genre examples.")

    # Decoding arguments
    parser.add_argument("--top-n-styles-to-change", type=int, default=3)
    parser.add_argument(
        "--decoding-strategy",
        type=str,
        default="default",
        choices=["default", "temperature", "greedy", "top_k", "top_p", "mixed"],
    )
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--use-constrained-decoding", action="store_true")
    parser.add_argument("--use-random-directions", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--direction-seed",
        type=int,
        default=None,
        help="Seed for style-axis selection; defaults to --seed.",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="Seed for prompt construction and LLM decoding; defaults to --seed.",
    )
    parser.add_argument("--max-gpu-memory-utilization", type=float, default=0.95)

    # Style transfer arguments
    parser.add_argument("--style-transfer", action="store_true")
    parser.add_argument(
        "--target-author-input-file", type=str, default="target_author_data.jsonl"
    )

    args = parser.parse_args()
    pprint.pprint(vars(args))
    direction_seed = (
        args.direction_seed if args.direction_seed is not None else args.seed
    )
    generation_seed = (
        args.generation_seed if args.generation_seed is not None else args.seed
    )
    random.seed(generation_seed)
    np.random.seed(direction_seed)

    data = pd.read_json(args.input_file, lines=True).drop(
        columns=[args.output_key, "StyleRemix_directions"], errors="ignore"
    )
    author_data = data.explode(args.author_key, ignore_index=True)
    author_texts = author_data.groupby(args.author_key)[args.text_key].apply(list)

    if args.style_transfer:
        target_author_data = pd.read_json(args.target_author_input_file, lines=True)
        target_author_texts = pd.Series(
            {"target_author": target_author_data[args.text_key].tolist()}
        )
        author_texts = pd.concat([author_texts, target_author_texts])

    with open(args.config_file, "r") as f:
        styleremix_config = json.load(f)

    evaluation_runner = EvaluationRunner(
        classifier_config=styleremix_config["classifiers"],
        lang=args.lang,
        library=args.library,
    )
    all_author_scores = {}
    for author_id, texts in author_texts.items():
        author_scores = evaluation_runner(texts)
        all_author_scores[author_id] = author_scores
    evaluation_runner.cleanup()

    author_scores = pd.DataFrame.from_dict(all_author_scores, orient="index")
    normalized_scores = (author_scores / author_scores.max(axis=0)).fillna(0)
    score_mean = normalized_scores.mean(axis=0)
    score_std = normalized_scores.std(axis=0)
    
    if args.lang in ["zh", "ar"]:
        types = [
            "persuasive_more",
            "narrative_more",
            "expository_more",
            "descriptive_more",
        ]
    elif args.lang in ["en", "ru"]:
        types = [
            "type_persuasive",
            "type_narrative",
            "type_expository",
            "type_descriptive"
        ]
        
    all_style_axes = list(normalized_scores.columns)
    available_types = [type_name for type_name in types if score_std[type_name] > 0]

    if args.style_transfer:
        author_directions = choose_directions_target_author(
            normalized_scores,
            args.top_n_styles_to_change,
            available_types,
            types,
            score_std,
            target_author_id="target_author",
            mix_toward=True,
        )
    else:
        if args.use_random_directions:
            print("Choosing random directions for target author")
            author_directions = choose_random_directions(
                normalized_scores,
                args.top_n_styles_to_change,
                types,
                all_style_axes,
                mix_toward=True,
            )
            print(author_directions)
        else:
            author_directions = choose_sliders_round_robin(
                normalized_scores,
                args.top_n_styles_to_change,
                available_types,
                types,
                score_std,
                closest=False,
            )
        
    author_directions.name = "StyleRemix_directions"
    author_data = author_data.merge(
        author_directions, left_on=args.author_key, right_index=True
    )
    combine_directions_partial = partial(combine_directions, all_types=types)
    agg_rules = {
        args.author_key: list,
        **{
            col: "first"
            for col in author_data.columns
            if col not in [args.document_key, args.author_key, author_directions.name]
        },
        author_directions.name: combine_directions_partial,
    }
    author_data = author_data.groupby(args.document_key).agg(agg_rules).reset_index()

    # Process genre input if present
    genre_examples = None
    if args.genre_input_file and os.path.exists(args.genre_input_file):
        genre_data = pd.read_json(args.genre_input_file, lines=True)
        # Droping NaNs just to be safe
        genre_examples = genre_data[args.text_key].dropna().tolist()

    if not args.prompt_remix:
        remix_runner = RemixRunner(
            model_id=args.model_id,
            adapter_paths=styleremix_config["adapters"],
            lang=args.lang,
        )
        tqdm.pandas(desc="Processing texts", total=len(author_data))
        author_data[args.output_key] = author_data.progress_apply(
            lambda row: remix_runner.remix(
                row[args.text_key], row["StyleRemix_directions"]
            ),
            axis=1,
        )
    else:
        config_file_dir = os.path.abspath(os.path.dirname(args.config_file))
        with open(
            os.path.join(config_file_dir, f"prompt_dict_{args.lang}.toml"), "rb"
        ) as f:
            prompt_dict = tomli.load(f)

        prompt_remix_runner = PromptRemixRunner(
            model_id=args.model_id,
            prompt_dict=prompt_dict,
            lang=args.lang,
            decoding_strategy=args.decoding_strategy,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_gpu_memory_utilization=args.max_gpu_memory_utilization,
            seed=generation_seed,
        )
        author_data[args.output_key] = prompt_remix_runner.remix(
            author_data[args.text_key].tolist(),
            author_data[author_directions.name].tolist(),
            use_constrained_decoding=args.use_constrained_decoding,
            genre_examples=genre_examples # Pass parsed examples down to the runner
        )

    author_data.to_json(args.output_file, lines=True, orient="records", force_ascii=False)


if __name__ == "__main__":
    main()