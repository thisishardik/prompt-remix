
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from scripts.utils.batch_translation import generate_output_path, translate_hrs

def test_generate_output_path(tmp_path):
    filepath = tmp_path / "en" / "some_file_english.jsonl"
    output_path = generate_output_path(filepath, source="en", target="zh")
    
    expected_dir = tmp_path / "zh" / "translated" / "en"
    assert output_path.parent == expected_dir
    assert output_path.name == "some_file_chinese.jsonl"
    assert expected_dir.is_dir()


@patch("scripts.utils.batch_translation.Translator")
def test_translate_hrs(mock_translator, fs): # fs is the pyfakefs fixture
    mock_translator_instance = MagicMock()
    mock_translator_instance.translate.side_effect = lambda text: f"translated_{text}"
    mock_translator.return_value = mock_translator_instance

    source_file = Path("/gscratch/ifml1/hiatus/hrs/queries/en/test_english.jsonl")
    fs.create_file(
        source_file, 
        contents=json.dumps({"fullText": "This is a test with <PERSON>."}) + "\n"
    )

    translate_hrs(source_file, source="en", target="zh", backend="google")

    output_path = generate_output_path(source_file, source="en", target="zh")
    assert fs.exists(output_path)
    
    with open(output_path, "r") as f:
        line = f.readline()
        data = json.loads(line)
        assert data["fullText"] == "translated_This is a test with Mika.".replace("米卡 ", "<PERSON>")
