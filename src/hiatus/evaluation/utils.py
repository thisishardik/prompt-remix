import re
import spacy
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from typing import Any, Callable, List, Tuple
from datasets import Dataset
from functools import partial

SPACY_LANG_MODELS = {
    'ru': 'ru_core_news_lg',
    'en': 'en_core_web_lg',
    'zh': 'zh_core_web_lg'
}

def load_model_and_tokenizer(model_cls: type, args) -> Tuple[AutoModel, AutoTokenizer]:
    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "max_memory": max_memory,
        "device_map": device_map
    }
    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if args.low_bit:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)

    model = model_cls.from_pretrained(args.model, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, model_max_length=args.max_length)

    if not tokenizer.pad_token_id:
        if model.config.pad_token_id is not None:
            tokenizer.pad_token_id = model.config.pad_token_id
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else: 
            tokenizer.pad_token_id = model.config.eos_token_id

    if model_cls == AutoModelForCausalLM:
        tokenizer.padding_side = "left"

    return model, tokenizer

def split_sentences(text: str, nlp) -> List[Tuple[int, str]]:
    # Output format: (sentence_id, sentence_text)
    doc = nlp(text)
    return [(i, sent.text.strip()) for i, sent in enumerate(doc.sents)]

def sentence_process(
    data: Dataset, 
    input_key: str, 
    output_key: str, 
    lang: str, 
    num_workers: int,
    processor: Callable[[List[str]], List[Any]],
    reducer: Callable[[List[Any]], Any]= " ".join
) -> Dataset:
    nlp = spacy.load(SPACY_LANG_MODELS[lang])
    sent_splitter = partial(split_sentences, nlp=nlp)
    
    clean_input_key = f"{input_key}###clean"
    dataset = data.map(lambda example, idx: {
        "idx": idx,
        clean_input_key: re.sub(r'(\w)(\n)', r'\1.\2', example[input_key])
    }, num_proc=num_workers, with_indices=True)
    
    pdf = dataset.to_pandas()
    pdf['sentences'] = pdf[clean_input_key].swifter.apply(sent_splitter)
    pdf = pdf.explode('sentences')
    pdf['sentence_id'], pdf['sentence_text'] = zip(*pdf['sentences'])
    pdf = pdf.drop(columns=['sentences'])
    pdf['processed'] = processor(pdf['sentence_text'].tolist())
    pdf = pdf.sort_values(by=['idx', 'sentence_id']).groupby('idx', sort=False)['processed'].apply(reducer).reset_index()
    
    result = pdf['processed'].tolist()   
    dataset = dataset.remove_columns(['idx', clean_input_key])
    if output_key in dataset.column_names:
        dataset = dataset.remove_columns(output_key)
    dataset = dataset.add_column(output_key, result)
    return dataset
