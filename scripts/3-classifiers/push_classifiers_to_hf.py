import argparse
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hiatus.config import hf_username

def main(args: argparse.Namespace) -> None:
    """Locate classifier folders and push them to the Hugging Face Hub."""
    model_name = args.model_name
    classifiers_paths = {path.name.split("_")[0]: path for path in Path(args.classifiers_dir).glob("*_classifier") }
    for style_axis, model_path in classifiers_paths.items():
        print(f"Pushing {style_axis} classifier.")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        hf_model_name = f"{hf_username}/{style_axis}-classifier-{model_name}-data"
        model.push_to_hub(hf_model_name, private=True)
        tokenizer.push_to_hub(hf_model_name, private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push classifiers to Hugging Face Hub")
    parser.add_argument(
        "--classifiers-dir",
        type=str,
        default="/gscratch/stf/mpotto/hiatus/classifiers/zh",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="chinese-roberta-wwm-ext-large",
        help="Base model name",
    )
    args = parser.parse_args()
    main(args)
