import os
import pprint
import re
from argparse import ArgumentParser
from functools import partial
from typing import List

import torch
torch.load = partial(torch.load, weights_only=False)

import pymorphy3
import stanza
from datasets import load_dataset
from tqdm import tqdm

CASE_MAP = {
    "Nom": "nomn",
    "Gen": "gent",
    "Dat": "datv",
    "Acc": "accs",
    "Ins": "ablt",
    "Loc": "loct",
    "Voc": "voct",
}


def run_pii_insert(texts: List[str], args) -> List[str]:
    """Run PII insertion over a list of texts according to `args`."""
    if args.lang == "ru":
        pii_processor = partial(
            ru_pii_insert,
            pii_token=args.pii_token,
            name=args.name,
            nlp=stanza.Pipeline("ru", processors="tokenize,pos,lemma,depparse"),
            morph=pymorphy3.MorphAnalyzer(),
        )
    elif args.lang == "en":
        pii_processor = partial(en_pii_insert, pii_token=args.pii_token, name=args.name)
    elif args.lang == "zh":
        pii_processor = partial(zh_pii_insert, pii_token=args.pii_token, name=args.name)
    elif args.lang == "ar":
        pii_processor = partial(ar_pii_insert, pii_token=args.pii_token, name=args.name)

    results = []
    for text in tqdm(texts, desc="Processing"):
        results.append(pii_processor(text))

    return results


def ru_pii_insert(text: str, pii_token: str, nlp, morph, name: str = "Саша") -> str:
    """Insert PII in Russian text, inflecting names to match grammatical case."""
    text = re.sub(r"(\w)(\n)", r"\1.\2", text)
    doc = nlp(text)
    pii_token_idxs = [
        [tok.id[0] for tok in sent.tokens if tok.text == pii_token]
        for sent in doc.sentences
    ]

    text_mod = text.replace(pii_token, name)
    doc_mod = nlp(text_mod)

    text_final = ""
    for sent, pii_idxs in zip(doc_mod.sentences, pii_token_idxs):
        for tok in sent.tokens:
            if tok.id[0] in pii_idxs:
                # Inflect to correct case if not nominative
                word = tok.words[0]
                if word.lemma == name and (word.upos == "NOUN" or word.upos == "PROPN"):
                    # 'Animacy=Inan|Case=Nom|Gender=Fem|Number=Sing' -- turn into dict
                    feats = dict([f.split("=") for f in word.feats.split("|")])
                    case = feats.get("Case", "Nom")
                    if case != "Nom":
                        name_inflected = (
                            morph.parse(name)[0]
                            .inflect({CASE_MAP[case]})
                            .word.capitalize()
                        )
                        text_final += name_inflected
                    else:
                        text_final += name
            else:
                text_final += tok.text

            text_final += tok.spaces_after

    return text_final


def en_pii_insert(text: str, pii_token: str, name: str = "Alex") -> str:
    """Insert PII token replacement for English by simple string replace."""
    text_final = text.replace(pii_token, name)
    return text_final


def zh_pii_insert(text: str, pii_token: str, name: str = "李晓明") -> str:
    """Insert PII token replacement for Chinese by simple string replace."""
    text_final = text.replace(pii_token, name)
    return text_final


def ar_pii_insert(text: str, pii_token: str, name: str = "نور") -> str:
    text_final = text.replace(pii_token, name)
    return text_final


def main() -> None:
    """CLI entrypoint: load data, run PII insertion, and save results."""
    
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--input-key", type=str, default="fullText")
    parser.add_argument("--output-key", type=str, default="piiReplacedText")
    parser.add_argument("--pii-token", type=str, default="<PERSON>")
    parser.add_argument("--name", type=str, default="Alex")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sampled-data-filepath", type=str, default=None, help="Path to JSON file with document IDs to sample.")
    args = parser.parse_args()
    pprint.pprint(vars(args))

    data = load_dataset("json", data_files=args.input_path)["train"]

    if args.sampled_data_filepath and os.path.exists(args.sampled_data_filepath):
        import pandas as pd
        sample_df = pd.read_json(args.sampled_data_filepath)
        sampled_doc_ids = set(sample_df['documentID'].values.tolist())
        data = data.filter(lambda x: x["documentID"] in sampled_doc_ids)
        print(f"Filtered dataset to {len(data)} examples based on {args.sampled_data_filepath}")

    results = run_pii_insert(data[args.input_key], args)

    if args.output_key in data.column_names:
        data = data.remove_columns(args.output_key)
    data = data.add_column(args.output_key, results)

    output_dir = args.output_path[: args.output_path.rfind("/") + 1]
    os.makedirs(output_dir, exist_ok=True)

    data.to_json(args.output_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
