import re
import pprint
from argparse import ArgumentParser
from functools import partial
from typing import List

import stanza
from datasets import load_dataset
from tqdm import tqdm

import torch
torch.load = partial(torch.load, weights_only=False)

def run_pii_remove(texts: List[str], orig_texts: List[str], args) -> List[str]:
    """Run PII removal over a list of texts according to `args`."""
    if args.lang == "ru":
        pii_processor = partial(
            ru_pii_remove,
            name=args.name,
            pii_token=args.pii_token,
            nlp=stanza.Pipeline("ru", processors="tokenize,pos,lemma,depparse"),
        )
    elif args.lang == "en":
        pii_processor = partial(en_pii_remove, name=args.name, pii_token=args.pii_token)
    elif args.lang == "zh":
        pii_processor = partial(zh_pii_remove, name=args.name, pii_token=args.pii_token)
    elif args.lang == "ar":
        pii_processor = partial(ar_pii_remove, name=args.name, pii_token=args.pii_token)
    else:
        raise ValueError

    results = []
    for text, orig_text in tqdm(zip(texts, orig_texts), desc="Processing", total=len(texts)):
        if args.pii_token in orig_text:
            results.append(pii_processor(text))
        else:
            results.append(text)

    return results


def ru_pii_remove(text: str, name: str, pii_token: str, nlp) -> str:
    """Remove PII in Russian text by replacing named entities with token."""
    text = re.sub(r"(\w)(\n)", r"\1.\2", text)
    doc = nlp(text)
    new_text = ""
    for sent in doc.sentences:
        for tok in sent.tokens:
            word = tok.words[0]
            if word.lemma == name and (word.upos == "NOUN" or word.upos == "PROPN"):
                new_text += pii_token
            else:
                new_text += tok.text
            new_text += tok.spaces_after

    return new_text


def en_pii_remove(text: str, name: str, pii_token: str) -> str:
    """Remove PII in English text by simple string replace of the name."""
    text_final = text.replace(name, pii_token)
    return text_final


def zh_pii_remove(text: str, name: str = "晨", pii_token: str = "<PERSON>") -> str:
    """Remove PII in Chinese text by simple string replace of the name."""
    text_final = text.replace(name, pii_token)
    return text_final


def ar_pii_remove(text: str, name: str = "القائد", pii_token: str = "<PERSON>") -> str:
    text_final = text.replace(name, pii_token)
    return text_final


def main() -> None:
    """CLI entrypoint: load data, remove PII, and save results."""
    
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--input-key", type=str, default="fullText")
    parser.add_argument("--output-key", type=str, default="piiReplacedText")
    parser.add_argument("--original-key", type=str, default="fullText", help="Key for the original text to check if PII was inserted.")
    parser.add_argument("--pii-token", type=str, default="<PERSON>")
    parser.add_argument("--name", type=str, default="Alex")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    pprint.pprint(vars(args))

    data = load_dataset("json", data_files=args.input_path)["train"]
    orig_texts = data[args.original_key] if args.original_key in data.column_names else [""] * len(data)
    results = run_pii_remove(data[args.input_key], orig_texts, args)
    
    if args.output_key in data.column_names:
        data = data.remove_columns(args.output_key)
    
    data = data.add_column(args.output_key, results)
    data.to_json(args.output_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
