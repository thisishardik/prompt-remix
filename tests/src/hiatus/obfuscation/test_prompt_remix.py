
import pytest
from unittest.mock import patch, MagicMock
from src.hiatus.obfuscation.prompt_remix import PromptRemixRunner

@pytest.fixture
def mock_vllm_and_tokenizer():
    with patch("vllm.LLM") as mock_llm, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
        
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.pad_token = "<pad>"
        mock_tokenizer.return_value.eos_token = "<eos>"
        
        yield mock_llm, mock_tokenizer

@pytest.fixture
def runner(mock_vllm_and_tokenizer):
    prompt_dict = {
        "preamble": "Rewrite the text.",
        "postamble": "Ensure it is good.",
        "length_less": "Make it shorter.",
        "formality_more": "Make it more formal with ##style_strength## strength.",
        "type_narrative": "Make it a narrative."
    }
    return PromptRemixRunner(
        model_id="dummy_model",
        prompt_dict=prompt_dict,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        lang="en",
    )

def test_get_style_strength(runner):
    assert runner._get_style_strenght("en") is not None
    assert runner._get_style_strenght("ru") is not None
    assert runner._get_style_strenght("zh") is not None
    assert runner._get_style_strenght("ar") is not None
    with pytest.raises(ValueError):
        runner._get_style_strenght("fr")

def test_normalize_directions(runner):
    directions = {"length_more": -0.5, "formality_more": 0.7, "type_narrative": 1.0, "zero_style": 0}
    norm_directions = runner.normalize_directions(directions)
    assert "length_less" in norm_directions
    assert norm_directions["length_less"] == 0.5
    assert "formality_more" in norm_directions
    assert norm_directions["formality_more"] == 0.7
    assert "type_narrative" in norm_directions
    assert norm_directions["type_narrative"] == 1.0
    assert "zero_style" not in norm_directions

def test_craft_instruction(runner):
    directions = {"formality_more": 0.9, "length_less": 0.5}
    instruction = runner.craft_instruction(directions)
    assert "Rewrite the text." in instruction
    assert "Make it more formal with strong strength." in instruction
    assert "Make it shorter." in instruction
    assert "Ensure it is good." in instruction

def test_postprocess_completion_llama(runner):
    text = "assistant\n\nThis is the assistant's response."
    processed_text = runner.postprocess_completion_llama(text)
    assert processed_text == "This is the assistant's response."

def test_postprocess_completion_qwen(runner):
    text = "some thoughts </think> This is the actual response."
    processed_text = runner.postprocess_completion_qwen(text)
    assert processed_text == "This is the actual response."

def test_build_sampling_params(runner):
    runner.decoding_strategy = "top_k"
    runner._build_sampling_params()
    params = runner.sampling_params
    assert params.top_k == 50
    assert params.top_p == 1.0
    assert params.temperature == 1.0

    runner.decoding_strategy = "greedy"
    runner._build_sampling_params()
    params = runner.sampling_params
    assert params.temperature == 0.0
    assert params.top_k == 1
