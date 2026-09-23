
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from src.hiatus.obfuscation.evaluation import (
    SentenceLenghtEvaluator,
    FunctionWordEvaluator,
    GradeLevelEvaluator,
    ClassifierEvaluator,
    MultiClassEvaluator,
    EvaluationRunner,
    NLProcessor
)

@pytest.fixture
def mock_nlp_processor():
    processor = MagicMock(spec=NLProcessor)
    processor.sents.return_value = ["This is a sentence.", "This is another one."]
    processor.pos_tags.return_value = ["PRON", "VERB", "DET", "NOUN", "PUNCT", "PRON", "VERB", "DET", "ADJ", "NOUN", "PUNCT"]
    processor.function_word_pos_tags = {"PRON", "DET"}
    processor.language = "en"
    return processor

def test_sentence_length_evaluator(mock_nlp_processor):
    evaluator = SentenceLenghtEvaluator(language="en", library="spacy")
    evaluator.nlp = mock_nlp_processor
    result = evaluator(["some text"])
    # "This is a sentence." -> 4 words, "This is another one." -> 4 words
    assert result["length_more"] == pytest.approx(4.0)
    
def test_function_word_evaluator(mock_nlp_processor):
    evaluator = FunctionWordEvaluator(language="en", library="spacy")
    evaluator.nlp = mock_nlp_processor
    # 4 function words (PRON, DET, PRON, DET) out of 11 tokens, but total_word_count is based on isalpha()
    text = "he is a boy. she is a good girl."
    result = evaluator([text])
    # function_word_count = 4, total_word_count = 25
    assert result["function_more"] == pytest.approx(4/25)


@patch("textstat.flesch_kincaid_grade", return_value=10.0)
@patch("textstat.linsear_write_formula", return_value=12.0)
@patch("textstat.gunning_fog", return_value=11.0)
def test_grade_level_evaluator(mock_gunning, mock_linsear, mock_fk, mock_nlp_processor):
    evaluator = GradeLevelEvaluator(language="en", library="spacy")
    evaluator.nlp = mock_nlp_processor
    result = evaluator(["some text"])
    assert result["grade_more"] == pytest.approx(11.0)


@patch("transformers.pipeline")
@patch("transformers.AutoModelForSequenceClassification")
@patch("transformers.AutoTokenizer")
def test_classifier_evaluator(mock_tok, mock_model, mock_pipeline):
    mock_classifier = MagicMock()
    mock_classifier.return_value = [{"label": "positive", "score": 0.9}]
    mock_pipeline.return_value = mock_classifier
    
    # Mock the model's config
    mock_model.from_pretrained.return_value.config.num_labels = 2

    evaluator = ClassifierEvaluator(model_name_or_path="dummy", label="positive")
    result = evaluator(["a positive text"])
    assert result["positive"] == 0.9

@patch("transformers.pipeline")
@patch("transformers.AutoModelForSequenceClassification")
@patch("transformers.AutoTokenizer")
def test_multiclass_evaluator(mock_tok, mock_model, mock_pipeline):
    mock_classifier = MagicMock()
    mock_classifier.return_value = [[{"label": "class_A", "score": 0.8}]]
    mock_pipeline.return_value = mock_classifier
    
    # Mock the model's config
    model_mock = MagicMock()
    model_mock.config.id2label = {0: "class_A", 1: "class_B"}
    mock_model.from_pretrained.return_value = model_mock

    evaluator = MultiClassEvaluator(model_name_or_path="dummy", label="*")
    result = evaluator(["some text"])

    assert "class_A" in result
    assert "class_B" in result
    assert result["class_A"] == pytest.approx(0.8)
    assert result["class_B"] == 0.0

def test_evaluation_runner():
    evaluator1 = MagicMock()
    evaluator1.return_value = {"metric1": 0.5}
    evaluator2 = MagicMock()
    evaluator2.return_value = {"metric2": 0.8}

    with patch.object(EvaluationRunner, "__init__", lambda self, classifier_config, lang, library: None):
        runner = EvaluationRunner({}, "en", "spacy")
        runner.evaluators = [evaluator1, evaluator2]
        results = runner(["some text"])

    assert results == {"metric1": 0.5, "metric2": 0.8}
    evaluator1.assert_called_once_with(["some text"])
    evaluator2.assert_called_once_with(["some text"])

