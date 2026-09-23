
import pytest
from unittest.mock import patch, MagicMock

from unittest.mock import patch
import sys
sys.modules['hiatus.config'] = MagicMock()

from src.hiatus.translation.translate import Translator


@patch("deepl.DeepLClient")
def test_translator_deepl(mock_deepl_client):
    mock_instance = mock_deepl_client.return_value
    mock_instance.translate_text.return_value.text = "translated text"
    
    translator = Translator(source="en", target="ru", backend="deepl")
    result = translator.translate("original text")
    
    assert result == "translated text"
    mock_deepl_client.assert_called_once() # Check that the client was initialized
    mock_instance.translate_text.assert_called_with(
        "original text",
        source_lang="EN",
        target_lang="RU"
    )

@patch("deep_translator.GoogleTranslator")
def test_translator_google(mock_google_translator):
    mock_instance = mock_google_translator.return_value
    mock_instance.translate.return_value = "translated text"

    translator = Translator(source="en", target="ar", backend="google")
    result = translator.translate("original text")

    assert result == "translated text"
    mock_google_translator.assert_called_with(source="en", target="ar")
    mock_instance.translate.assert_called_with("original text")

def test_translator_invalid_backend():
    with pytest.raises(AttributeError):
        translator = Translator(source="en", target="ru", backend="invalid")
        translator.translate("text")

