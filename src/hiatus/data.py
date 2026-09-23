from pathlib import Path
from typing import Literal, Optional, Dict

from hiatus.config import common_storage_dir, privatized_dir, data_dir

language_lookup = {
    "en": "english",
    "ru": "russian",
    "zh": "chinese",
    "ar": "arabic",
}

length_lookup = {
    "en_short_crossGenre": "short",
    "zh_short_crossGenre": "short",
    "zh_medium_crossGenre": "medium",
    "perGenre-HRS3.1": "short",
    "perGenre-HRS3.2": "short",
    "perGenre-HRS3.3": "short",
    "perGenre-HRS3.4": "short",
    "perGenre-HRS3.5": "short",
    "perGenre-HRS3.301": "long",
    "perGenre-HRS3.302": "medium",
    "perGenre-HRS3.303": "short",
    "perGenre-HRS3.304": "medium",
    "perGenre-HRS3.305": "short",
}

phase_lookup = {
    "AMINA": "3",
    "bbn_aljazeera_blog_1": "3",
    "bbn_ida2at": "3",
    "kaggle_arabic_news_1": "3",
    "ar_twitter_1": "3",
    "saudinewsnet": "3",
    "bbn_aljazeera_blog_1": "3",
    "kaggle_arabic_news_1": "3",
    "AMINA": "3",
    "ru_crossGenre": "2",
    "en_crossGenre": "2",
    "en_short_crossGenre": "3",
    "zh_short_crossGenre": "3",
    "zh_medium_crossGenre": "3",
    "perGenre-HRS2.1": "2",
    "perGenre-HRS2.2": "2",
    "perGenre-HRS2.3": "2",
    "perGenre-HRS2.4": "2",
    "perGenre-HRS2.5": "2",
    "perGenre-HRS3.1": "3",
    "perGenre-HRS3.2": "3",
    "perGenre-HRS3.3": "3",
    "perGenre-HRS3.4": "3",
    "perGenre-HRS3.5": "3",
    "perGenre-HRS2.101": "2",
    "perGenre-HRS2.102": "2",
    "perGenre-HRS2.103": "2",
    "perGenre-HRS2.104": "2",
    "perGenre-HRS2.105": "2",
    "perGenre-HRS3.301": "3",
    "perGenre-HRS3.302": "3",
    "perGenre-HRS3.303": "3",
    "perGenre-HRS3.304": "3",
    "perGenre-HRS3.305": "3",
}


def hrs_paths(
    genre: Literal[
        "AMINA",  # non-hrs genre
        "bbn_aljazeera_blog_1",  # non-hrs genre
        "bbn_ida2at",  # non-hrs genre
        "kaggle_arabic_news_1",  # non-hrs genre
        "ar_twitter_1",  # non-hrs genre
        "saudinewsnet",  # non-hrs genre
        "ru_crossGenre",  # Phase 2
        "en_crossGenre",  # Phase 2
        "en_short_crossGenre",  # Phase 3
        "zh_short_crossGenre",  # Phase 3
        "zh_medium_crossGenre",  # Phase 3
        "perGenre-HRS2.1",  # english
        "perGenre-HRS2.2",  # english
        "perGenre-HRS2.3",  # english
        "perGenre-HRS2.4",  # english
        "perGenre-HRS2.5",  # english
        "perGenre-HRS3.1",  # english short
        "perGenre-HRS3.2",  # english short
        "perGenre-HRS3.3",  # english short
        "perGenre-HRS3.4",  # english short
        "perGenre-HRS3.5",  # english short
        "perGenre-HRS2.101",  # russian
        "perGenre-HRS2.102",  # russian
        "perGenre-HRS2.103",  # russian
        "perGenre-HRS2.104",  # russian
        "perGenre-HRS2.105",  # russian
        "perGenre-HRS3.301",  # chinese long
        "perGenre-HRS3.302",  # chinese medium
        "perGenre-HRS3.303",  # chinese short
        "perGenre-HRS3.304",  # chinese medium
        "perGenre-HRS3.305",  # chinese short
    ],
    language: Literal["en", "ru", "zh", "ar"] = "zh",
    pipeline_suffix: Optional[str] = "prompt-remix",
) -> Dict:

    if language == "en":
        assert genre in [
            "en_short_crossGenre",
            "en_crossGenre",
            "perGenre-HRS2.1",
            "perGenre-HRS2.2",
            "perGenre-HRS2.3",
            "perGenre-HRS2.4",
            "perGenre-HRS2.5",
            "perGenre-HRS3.1",  # english short
            "perGenre-HRS3.2",  # english short
            "perGenre-HRS3.3",  # english short
            "perGenre-HRS3.4",  # english short
            "perGenre-HRS3.5",  # english short
        ]

    elif language == "ru":
        assert genre in [
            "ru_crossGenre",
            "perGenre-HRS2.101",
            "perGenre-HRS2.102",
            "perGenre-HRS2.103",
            "perGenre-HRS2.104",
            "perGenre-HRS2.105",
        ]
    elif language == "zh":
        assert genre in [
            "zh_short_crossGenre",
            "zh_medium_crossGenre",
            "perGenre-HRS3.301",  # chinese long
            "perGenre-HRS3.302",  # chinese medium
            "perGenre-HRS3.303",  # chinese short
            "perGenre-HRS3.304",  # chinese medium
            "perGenre-HRS3.305",  # chinese short
        ]
    elif language == "ar":
        assert genre in [
            "AMINA", 
            "bbn_aljazeera_blog_1",
            "bbn_ida2at",
            "kaggle_arabic_news_1",
            "ar_twitter_1",
            "saudinewsnet",
        ]
    else:
        raise ValueError(
            f"Invalid combination of language ({language}) and genre ({genre})."
        )

    phase = phase_lookup[genre]

    if phase == "2":
        assert language in ["en", "ru"]

        hrs_dir = Path(common_storage_dir, f"hrs{phase}")

        if "crossGenre" in genre:
            genre = "crossGenre"
            base_filename = f"hrs_06-27-24_{language_lookup[language]}_{genre}-combined"
        else:
            base_filename = f"hrs_06-27-24_{language_lookup[language]}_{genre}"

        base_dir = Path(hrs_dir, f"mode_{genre}")

        paths = {
            "TA1": {
                "data": Path(base_dir, "TA1", base_filename, "data"),
                "groundtruth": Path(
                    base_dir,
                    "TA1",
                    base_filename,
                    "groundtruth",
                    f"{base_filename}_TA1_groundtruth.jsonl",
                ),
            },
            "TA3": {
                "data": {
                    "queries": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "data",
                        f"{base_filename}_TA3_input_queries.jsonl",
                    ),
                    "candidates": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "data",
                        f"{base_filename}_TA3_input_candidates.jsonl",
                    ),
                    "privatized": Path(
                        privatized_dir,
                        language,
                        f"{base_filename}_TA3_privatized_queries_{pipeline_suffix}.jsonl",
                    ),
                },
                "groundtruth": {
                    "candidate_labels": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "groundtruth",
                        f"{base_filename}_TA3_candidate-labels.txt",
                    ),
                    "groundtruth": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "groundtruth",
                        f"{base_filename}_TA3_groundtruth.npy",
                    ),
                    "query_labels": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "groundtruth",
                        f"{base_filename}_TA3_query-labels.txt",
                    ),
                },
            },
        }

    elif phase == "3" and language in ["zh", "en"]:

        hrs_dir = Path(common_storage_dir, f"hrs{phase}")

        base_dir = Path(
            hrs_dir,
            "HRS_evaluation_samples",
            f"HRS3_{language_lookup[language]}_{length_lookup[genre]}",
        )

        if genre == "zh_short_crossGenre":
            base_filename = f"HRS3_chinese_short_sample-0_crossGenre"
        elif genre == "zh_medium_crossGenre":
            base_filename = f"HRS3_chinese_medium_sample-0_crossGenre"
        elif genre == "en_short_crossGenre":
            base_filename = f"HRS3_english_short_sample-0_crossGenre"
        else:
            base_filename = f"HRS3_{language_lookup[language]}_{length_lookup[genre]}_sample-0_{genre}"

        paths = {
            "TA1": {
                "data": Path(base_dir, "TA1", base_filename, "data"),
                "groundtruth": Path(
                    base_dir,
                    "TA1",
                    base_filename,
                    "groundtruth",
                    f"{base_filename}_TA1_groundtruth.jsonl",
                ),
            },
            "TA3": {
                "data": {
                    "queries": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "data",
                        f"{base_filename}_TA3_input_queries.jsonl",
                    ),
                    "candidates": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "data",
                        f"{base_filename}_TA3_input_candidates.jsonl",
                    ),
                    "privatized": Path(
                        privatized_dir,
                        language,
                        f"{base_filename}_TA3_privatized_queries_{pipeline_suffix}.jsonl",
                    ),
                },
                "groundtruth": {
                    "candidate_labels": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "groundtruth",
                        f"{base_filename}_TA3_candidate-labels.txt",
                    ),
                    "groundtruth": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "groundtruth",
                        f"{base_filename}_TA3_groundtruth.npy",
                    ),
                    "query_labels": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "groundtruth",
                        f"{base_filename}_TA3_query-labels.txt",
                    ),
                },
            },
        }
    elif phase == "3" and language == "ar":

        nonhrs_dir = Path(common_storage_dir, f"nonhrs")

        base_dir = Path(
            nonhrs_dir,
            f"{genre}",
        )
        base_filename = f"{genre}"

        paths = {
            "TA1": {
                "data": Path(base_dir, "TA1", base_filename, "data"),
                "groundtruth": Path(
                    base_dir,
                    "TA1",
                    base_filename,
                    "groundtruth",
                    f"{base_filename}_TA1_groundtruth.jsonl",
                ),
            },
            "TA3": {
                "data": {
                    "queries": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "data",
                        f"{base_filename}_TA3_input_queries.jsonl",
                    ),
                    "candidates": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "data",
                        f"{base_filename}_TA3_input_candidates.jsonl",
                    ),
                    "privatized": Path(
                        privatized_dir,
                        language,
                        f"{base_filename}_TA3_privatized_queries_{pipeline_suffix}.jsonl",
                    ),
                },
                "groundtruth": {
                    "candidate_labels": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "groundtruth",
                        f"{base_filename}_TA3_candidate-labels.txt",
                    ),
                    "groundtruth": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "groundtruth",
                        f"{base_filename}_TA3_groundtruth.npy",
                    ),
                    "query_labels": Path(
                        base_dir,
                        "TA3",
                        base_filename,
                        "groundtruth",
                        f"{base_filename}_TA3_query-labels.txt",
                    ),
                },
            },
        }
    else:
        raise ValueError
    return paths

def translated_paths(
    input_path: str,
    source: str = "en",
    target: str = "ar",
) -> Dict:
    data_source_map = {"ar": "nonhrs", "zh": "hrs2", "en": "hrs2", "ru": "hrs2"}
    queries_path = Path(data_dir, f"{data_source_map[source]}/queries/")
    output_dir = Path(queries_path, f"{target}/translated/{source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = input_path.name.replace(
        language_lookup[source], language_lookup[target], 1
    )
    output_path = Path(output_dir, output_name)
    return output_path


# ────────────────────────────────────────────────────────────
# AuthBench paths
# ────────────────────────────────────────────────────────────

authbench_dir = Path(common_storage_dir, "data", "raw", "authbench")


def authbench_paths(
    language: str = "en",
    genre: Optional[str] = None,
    pipeline_suffix: Optional[str] = None,
) -> Dict:
    assert language in ("en", "ru", "zh", "ar"), f"Unsupported language: {language}"
    if pipeline_suffix is None:
        pipeline_suffix = f"promptremix_authbench-{language}0-3"

    lang_dir = authbench_dir / language

    if genre is not None:
        genre_safe = genre.replace("/", "_")
        queries_path = (
            lang_dir / "per_genre"
            / f"authbench_{language}_{genre_safe}_queries.jsonl"
        )
        gt_path = (
            lang_dir / "per_genre"
            / f"authbench_{language}_{genre_safe}_ground_truth.npy"
        )
    else:
        queries_path = lang_dir / f"authbench_{language}_queries.jsonl"
        gt_path = lang_dir / f"authbench_{language}_ground_truth.npy"

    candidates_path = lang_dir / f"authbench_{language}_candidates.jsonl"

    priv_suffix = f"_{genre_safe}" if genre else ""
    privatized_path = Path(
        privatized_dir,
        "authbench",
        language,
        f"authbench_{language}{priv_suffix}_privatized_queries_{pipeline_suffix}.jsonl",
    )

    return {
        "queries": queries_path,
        "candidates": candidates_path,
        "ground_truth": gt_path,
        "privatized": privatized_path,
    }


def blog_authorship_paths(
    sample_name: str = "sample_100",
    pipeline_suffix: Optional[str] = "promptremix-blog",
) -> Dict:
    base_dir = Path(common_storage_dir, "data", "raw", "blog_authorship", sample_name)
    data_dir = base_dir / "TA3" / "data"
    gt_dir = base_dir / "TA3" / "groundtruth"

    queries_path = data_dir / "input_queries.jsonl"
    candidates_path = data_dir / "input_candidates.jsonl"
    gt_path = gt_dir / "groundtruth.npy"

    privatized_path = Path(
        privatized_dir,
        "blog_authorship",
        sample_name,
        f"blog_en_privatized_queries_{pipeline_suffix}.jsonl",
    )

    return {
        "queries": queries_path,
        "candidates": candidates_path,
        "ground_truth": gt_path,
        "privatized": privatized_path,
    }

