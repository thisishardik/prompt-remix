
import pytest
from unittest.mock import patch, MagicMock
from src.hiatus.obfuscation.style_remix import RemixRunner

@pytest.fixture
def mock_peft_and_transformers():
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_base_model, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, \
         patch("peft.PeftModel.from_pretrained") as mock_peft_model:
        
        mock_tokenizer.return_value = MagicMock()
        mock_base_model.return_value = MagicMock()
        mock_peft_model.return_value = MagicMock()
        
        yield mock_base_model, mock_tokenizer, mock_peft_model

@pytest.fixture
def runner(mock_peft_and_transformers):
    adapter_paths = {"adapter1": "path1", "adapter2": "path2"}
    return RemixRunner(model_id="dummy_model", adapter_paths=adapter_paths, lang="en")

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

def test_craft_prompt(runner):
    text = "This is the original text."
    prompt = runner.craft_prompt(text)
    assert prompt == "### Original: This is the original text.\n ### Rewrite:"

    runner_ru = RemixRunner(model_id="dummy_model", adapter_paths={}, lang="ru")
    prompt_ru = runner_ru.craft_prompt(text)
    assert prompt_ru == "### Оригинал: This is the original text.\n ### Перепишите: "

def test_replace_adapters(runner):
    directions = {"formality_more": 0.9, "length_less": 0.5}
    adapter_name = "formality_more_90-length_less_50"
    
    runner.model.add_weighted_adapter = MagicMock()
    runner.model.set_adapter = MagicMock()

    runner.replace_adapters(directions)

    runner.model.add_weighted_adapter.assert_called_with(
        list(directions.keys()),
        weights=list(directions.values()),
        adapter_name=adapter_name,
        combination_type="cat"
    )
    runner.model.set_adapter.assert_called_with(adapter_name)
    assert runner.curr_adapter_name == adapter_name

    # Test that delete_adapter is called on the second run
    runner.model.delete_adapter = MagicMock()
    new_directions = {"formality_more": 0.5}
    new_adapter_name = "formality_more_50"
    runner.replace_adapters(new_directions)
    runner.model.delete_adapter.assert_called_with(adapter_name)
    assert runner.curr_adapter_name == new_adapter_name

