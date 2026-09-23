from typing import Literal

import deepl
from deep_translator import GoogleTranslator

from hiatus.config import deepl_api_key

deepl_languages = {
    "en": "EN",
    "ru": "RU",
    "zh": "ZH-HANS",
    "zh-Hant": "ZH-HANT",
    "ar": "AR",
}

google_languages = {"en": "en", "ru": "ru", "zh": "zh-CN", "ar": "ar"}


class Translator:

    def __init__(self, source: str, target: str, backend: Literal["deepl", "google"]):
        self.source = source
        self.target = target
        self.backend = backend

    def translate(self, text: str) -> str:
        if self.backend == "deepl":
            deepl_client = deepl.DeepLClient(deepl_api_key)
            translated = deepl_client.translate_text(
                text,
                source_lang=deepl_languages[self.source],
                target_lang=deepl_languages[self.target],
            ).text
        elif self.backend == "google":
            google_client = GoogleTranslator(
                source=google_languages[self.source], target=google_languages[self.target]
            )
            translated = ""
            try:
                translated = google_client.translate(text)
            except Exception as e:
                print(f"ERROR - [{e}] - [Translation failed for hence using DeepL backend] - [{text}]")
                deepl_client = deepl.DeepLClient(deepl_api_key)
                translated = deepl_client.translate_text(
                    text,
                    source_lang=deepl_languages[self.source],
                    target_lang=deepl_languages[self.target],
                ).text

        return translated
