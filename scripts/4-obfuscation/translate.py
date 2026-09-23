import argparse
import json
from pathlib import Path

from tqdm import tqdm

from hiatus.translation.translate import Translator
from hiatus.config import common_storage_dir, data_dir
from hiatus.data import translated_paths

language_lookup = {"zh": "chinese", "ar": "arabic", "en": "english", "ru": "russian"}
name_replacement_lookup = {"en": "Mika", "ru": "Мика", "zh": "米卡", "ar": "ميكا"}


# def generate_output_path_translate(
#     filepath: Path, source: str = "en", target: str = "zh"
# ) -> Path:
#     """Generate a canonical output Path for a translated queries file."""
#     queries_path = Path(data_dir, f"{data_source_map[source]}/queries/")
#     output_dir = Path(queries_path, f"{target}/translated/{source}")
#     output_dir.mkdir(parents=True, exist_ok=True)
#     output_name = filepath.name.replace(
#         language_lookup[source], language_lookup[target], 1
#     )
#     output_path = Path(output_dir, output_name)
#     return output_path


def translate(
    filepath: str,
    output_path: str,
    field: str = "fullText",
    source: str = "en",
    target: str = "zh",
    backtranslate: bool = False,
    backend: str = "google",
) -> None:
    """Translate entries in a jsonl file from `source` to `target`.

    Replaces <PERSON> tokens with language-specific placeholders during
    translation and restores them afterwards. Writes to a new file and
    reports how many entries were skipped.
    """
    filepath = Path(filepath)

    skipped = 0
    source_name = name_replacement_lookup[source]
    target_name = name_replacement_lookup[target]
    translator = Translator(source=source, target=target, backend=backend)

    if backtranslate:
        # Write to a temporary file in the same directory, then replace original atomically
        tmp_path = filepath.with_name(filepath.name + ".tmp")
        write_path = tmp_path
    else:
        write_path = output_path

    with (
        open(filepath, "r", encoding="utf-8") as infile,
        open(write_path, "w", encoding="utf-8") as outfile,
    ):
        for line in tqdm(infile):
            entry = json.loads(line)
            text = entry.get(field, "")

            # chunking
            max_char_limit = 200
            i = 0
            translated = ""

            for chunk_id, i in enumerate(range(0, len(text), max_char_limit)):
                print(f"Translating chunk #{chunk_id+1}")
                original_chunk = text[i:i+max_char_limit]
                translated_chunk = ""
                if "<PERSON>" in original_chunk:
                    temp_text = original_chunk.replace("<PERSON>", source_name)
                    translated_temp = translator.translate(temp_text)

                    if target_name not in translated_temp:
                        skipped += 1
                        continue
                    translated_chunk = translated_temp.replace(target_name, "<PERSON>")
                else:
                    translated_chunk = translator.translate(original_chunk)

                if translated_chunk == None:
                    translated_chunk = ""

                if source_name in translated_chunk:
                    skipped += 1
                    continue
                translated += translated_chunk

            entry[field] = translated
            entry["languages"] = [target]
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # if backtranslate, replace original with tmp file
    if backtranslate:
        # Path.replace() will atomically replace the destination if possible
        write_path.replace(filepath)
        final_path = filepath
    else:
        final_path = write_path

    print(final_path.as_posix(), skipped)


def main(args: argparse.Namespace) -> None:
    """CLI entrypoint: translate a file according to parsed CLI args."""
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    print(f"Running translation for {input_path}")

    translate(
        input_path,
        output_path,
        field=args.field,
        source=args.source,
        target=args.target,
        backtranslate=args.backtranslate,
        backend=args.backend,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="zh")
    parser.add_argument("--target", type=str, default="en")
    parser.add_argument("--field", type=str, default="fullText")
    parser.add_argument("--backtranslate", action="store_true")
    parser.add_argument("--input-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="/gscratch/ifml1/nonhrs/queries")
    parser.add_argument("--backend", type=str, default="google")
    args = parser.parse_args()
    main(args)
