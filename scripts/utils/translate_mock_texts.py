import json
from pathlib import Path

from deep_translator import GoogleTranslator
from tqdm import tqdm

from hiatus.config import data_paths

# Specify input and output file paths
input_path = Path(
    data_paths["evaluation"],
    "hrs_06-27-24_english_perGenre-HRS2.1_TA3_input_queries.jsonl",
)
output_path = Path(
    data_paths["evaluation"],
    "hrs_06-27-24_chinese_perGenre-HRS2.1_TA3_input_queries.jsonl",
)

# Initialize the translator (preserving PERSON tags)
translator = GoogleTranslator(source="en", target="zh-CN")

with (
    open(input_path, "r", encoding="utf-8") as infile,
    open(output_path, "w", encoding="utf-8") as outfile,
):
    for line in tqdm(infile):
        entry = json.loads(line)
        original_text = entry.get("fullText", "")

        # Only use placeholder logic if <PERSON> is present
        if "<PERSON>" in original_text:
            mock_name = "John"
            chinese_mock = "约翰"
            temp_text = original_text.replace("<PERSON>", mock_name)
            # Translate
            translated_temp = translator.translate(temp_text)
            # Check if the mock name was translated as expected
            if chinese_mock not in translated_temp:
                print(
                    f"WARNING: '{chinese_mock}' not found in translation. Line: {line.strip()}"
                )
            # Restore mock name (in Chinese) back to <PERSON>
            translated = translated_temp.replace(chinese_mock, "<PERSON>")
        else:
            # No <PERSON> tag, translate as usual
            translated = translator.translate(original_text)
        if mock_name in translated:
            print("Skipping entry. Mock English name detected in Chinese text.")
        else:
            entry["fullText"] = translated
            entry["languages"] = ["zh"]
            # Write entry to output
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
