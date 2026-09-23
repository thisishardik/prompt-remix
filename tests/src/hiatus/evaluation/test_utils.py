
import pytest
import spacy
from datasets import Dataset
from functools import partial

from src.hiatus.evaluation.utils import split_sentences, sentence_process

@pytest.fixture(scope="module")
def nlp_en():
    try:
        return spacy.load('en_core_web_lg')
    except OSError:
        spacy.cli.download('en_core_web_lg')
        return spacy.load('en_core_web_lg')

def test_split_sentences(nlp_en):
    text = "This is a sentence. This is another sentence."
    sentences = split_sentences(text, nlp_en)
    assert len(sentences) == 2
    assert sentences[0] == (0, "This is a sentence.")
    assert sentences[1] == (1, "This is another sentence.")

def test_sentence_process():
    data = Dataset.from_dict({"text": ["This is a sentence. This is another sentence."]})
    
    def dummy_processor(sentences):
        return [s.upper() for s in sentences]

    def dummy_reducer(processed_sentences):
        return " | ".join(processed_sentences)

    processed_dataset = sentence_process(
        data, 
        input_key="text", 
        output_key="processed_text", 
        lang="en", 
        num_workers=1,
        processor=dummy_processor,
        reducer=dummy_reducer
    )

    assert "processed_text" in processed_dataset.column_names
    assert processed_dataset["processed_text"][0] == "THIS IS A SENTENCE. | THIS IS ANOTHER SENTENCE."
