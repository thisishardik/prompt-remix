import argparse
import json
from pathlib import Path

from tqdm import tqdm

from hiatus.translation.translate import Translator

queries_path = Path("/gscratch/ifml1/hiatus/hrs/queries")
candidates_path = Path("/gscratch/ifml1/hiatus/hrs/candidates")

queries_path_en = [
    Path(queries_path, "en/hrs_06-27-24_english_crossGenre-combined_TA3_input_queries.jsonl"),
    Path(queries_path, "en/hrs_06-27-24_english_perGenre-HRS2.1_TA3_input_queries.jsonl"),
    Path(queries_path, "en/hrs_06-27-24_english_perGenre-HRS2.2_TA3_input_queries.jsonl"),
    Path(queries_path, "en/hrs_06-27-24_english_perGenre-HRS2.3_TA3_input_queries.jsonl"),
    Path(queries_path, "en/hrs_06-27-24_english_perGenre-HRS2.4_TA3_input_queries.jsonl"),
    Path(queries_path, "en/hrs_06-27-24_english_perGenre-HRS2.5_TA3_input_queries.jsonl"),
]

candidates_path_en = [
    Path(candidates_path, "en/hrs_06-27-24_english_crossGenre-combined_TA3_input_candidates.jsonl"),
    Path(candidates_path, "en/hrs_06-27-24_english_perGenre-HRS2.1_TA3_input_candidates.jsonl"),
    Path(candidates_path, "en/hrs_06-27-24_english_perGenre-HRS2.2_TA3_input_candidates.jsonl"),
    Path(candidates_path, "en/hrs_06-27-24_english_perGenre-HRS2.3_TA3_input_candidates.jsonl"),
    Path(candidates_path, "en/hrs_06-27-24_english_perGenre-HRS2.4_TA3_input_candidates.jsonl"),
    Path(candidates_path, "en/hrs_06-27-24_english_perGenre-HRS2.5_TA3_input_candidates.jsonl"),
]

queries_path_ru = [
    Path(queries_path, "ru/hrs_06-27-24_russian_crossGenre-combined_TA3_input_queries.jsonl"),
    Path(queries_path, "ru/hrs_06-27-24_russian_perGenre-HRS2.101_TA3_input_queries.jsonl"),
    Path(queries_path, "ru/hrs_06-27-24_russian_perGenre-HRS2.102_TA3_input_queries.jsonl"),
    Path(queries_path, "ru/hrs_06-27-24_russian_perGenre-HRS2.103_TA3_input_queries.jsonl"),
    Path(queries_path, "ru/hrs_06-27-24_russian_perGenre-HRS2.104_TA3_input_queries.jsonl"),
    Path(queries_path, "ru/hrs_06-27-24_russian_perGenre-HRS2.105_TA3_input_queries.jsonl"),
]

candidates_path_ru = [
    Path(candidates_path, "ru/hrs_06-27-24_russian_crossGenre-combined_TA3_input_candidates.jsonl"),
    Path(candidates_path, "ru/hrs_06-27-24_russian_perGenre-HRS2.101_TA3_input_candidates.jsonl"),
    Path(candidates_path, "ru/hrs_06-27-24_russian_perGenre-HRS2.102_TA3_input_candidates.jsonl"),
    Path(candidates_path, "ru/hrs_06-27-24_russian_perGenre-HRS2.103_TA3_input_candidates.jsonl"),
    Path(candidates_path, "ru/hrs_06-27-24_russian_perGenre-HRS2.104_TA3_input_candidates.jsonl"),
    Path(candidates_path, "ru/hrs_06-27-24_russian_perGenre-HRS2.105_TA3_input_candidates.jsonl"),
]

language_lookup = {"zh": "chinese", "ar": "arabic", "en": "english", "ru": "russian"}
name_replacement_lookup = {"en": "Mika", "ru": "Мика", "zh": "米卡 ", "ar": "ميكا"}


def generate_output_path(filepath, source="en", target="zh"):
    output_dir = Path(filepath.parent.parent, f"{target}/translated/{source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = Path(filepath.name).with_name(
        filepath.name.replace(f"{language_lookup[source]}", f"{language_lookup[target]}", 1)
    )
    output_path = Path(output_dir, output_name)
    return output_path


def translate_hrs(filepath, source="en", target="zh", backend="google"):
    output_path = generate_output_path(filepath, source=source, target=target)

    # Skip if already translated
    if output_path.exists():
        print(f"Skipping {filepath.name}, already translated.")
        return
    
    source_name = name_replacement_lookup[source]
    target_name = name_replacement_lookup[target]
    skipped = 0
    translator = Translator(source=source, target=target, backend=backend)

    with (
        open(filepath, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line in tqdm(infile):
            entry = json.loads(line)
            text = entry.get("fullText", "")

            if "<PERSON>" in text:
                temp_text = text.replace("<PERSON>", source_name)
                translated_temp = translator.translate(temp_text)

                if target_name not in translated_temp:
                    skipped += 1
                    continue
                translated = translated_temp.replace(target_name, "<PERSON>")
            else:
                translated = translator.translate(text)

            if source_name in translated:
                skipped += 1
                continue

            entry["fullText"] = translated
            entry["languages"] = [target]
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"{filepath.name}: Skipped {skipped}")


def main(args):
    if args.source == "en":
        queries = queries_path_en
        candidates = candidates_path_en
    elif args.source == "ru":
        queries = queries_path_ru
        candidates = candidates_path_ru
    else:
        raise ValueError("Unsupported source language")

    # Translate queries
    for path in queries:
        translate_hrs(path, source=args.source, target=args.target, backend=args.backend)

    # Translate candidates if requested
    if args.translate_candidates:
        for path in candidates:
            translate_hrs(path, source=args.source, target=args.target, backend=args.backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="en")
    parser.add_argument("--target", type=str, default="zh")
    parser.add_argument("--backend", type=str, default="google")
    parser.add_argument(
        "--translate-candidates", action="store_true",
        help="Also translate input_candidates files"
    )
    args = parser.parse_args()

    main(args)
