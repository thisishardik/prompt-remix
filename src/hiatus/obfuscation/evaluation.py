from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import spacy
import stanza
import torch
import textstat
from .readability_cn import ChineseReadability
from .readability_cn.nlp import JiebaNLP
from stanza.server import CoreNLPClient, StartServer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from hiatus.utils import extract_score

CORENLP_LANGUAGES = {
    "zh": "chinese",
    "ar": "arabic",
    "en": "english",
}

####################################
# Natural Language Processor
####################################

class NLProcessor:
    SPACY_MODELS = {
        "en": "en_core_web_lg",
        "ru": "ru_core_news_lg",
        "zh": "zh_core_web_lg",
    }

    SPACY_CONFIG = {
        "en": {},
        "ru": {},
        "zh": {"nlp.tokenizer.segmenter": "jieba"},
    }

    FUNCTION_WORD_POS = {
        "en": {"ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"},
        "ru": {"ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"},
        "zh": {"ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"},
        "ar": {
            "ADP",
            "AUX",
            "CCONJ",
            "DET",
            "NUM",
            "PART",
            "PRON",
            "SCONJ",
            "INTJ",
            "SYM",
        },
    }

    def __init__(
        self, language: str, library: Literal["spacy", "stanza", "corenlp"] = "spacy"
    ):
        self.language = language
        self.library = library
        self.nlp = self._load_nlp(library)
        self.function_word_pos_tags = self.FUNCTION_WORD_POS[language]

    def _load_nlp(self, library: Literal["spacy", "stanza", "corenlp"]):
        if library == "spacy":
            nlp = spacy.load(
                self.SPACY_MODELS[self.language], config=self.SPACY_CONFIG[self.language]
            )
        elif library == "stanza":
            nlp = stanza.Pipeline(lang=self.language, processors="tokenize,pos")
        elif library == "corenlp":
            assert self.language in CORENLP_LANGUAGES.keys()
            import socket

            def get_free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    return s.getsockname()[1]

            PORT = get_free_port()
            print(f"Using port {PORT}")

            try:
                nlp = CoreNLPClient(
                    endpoint=f"http://localhost:{PORT}",
                    annotators=["tokenize", "ssplit", "pos", "lemma"],
                    timeout=30000,
                    memory="6G",
                    be_quiet=True,
                    start_server=True,
                    # properties=CORENLP_LANGUAGES[self.language]
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to connect to CoreNLP server: {str(exc)}")
        else:
            raise ValueError(f"Unsupported library: {library}")

        return nlp

    def sents(self, text: str):
        if self.library == "spacy":
            doc = self.nlp(text)
            sents = [sent.text for sent in doc.sents]
        elif self.library == "stanza":
            doc = self.nlp(text)
            sents = [sent.text for sent in doc.sentences]
        elif self.library == "corenlp":
            doc = self.nlp.annotate(text)
            sents = [
                " ".join(token.word for token in sentence.token).strip()
                for sentence in doc.sentence
            ]
        return sents

    def pos_tags(self, text: str):
        if self.library == "spacy":
            doc = self.nlp(text)
            pos_tags = [token.pos_ for token in doc]
        elif self.library == "stanza":
            doc = self.nlp(text)
            pos_tags = [token.pos for sent in doc.sentences for token in sent.words]
        elif self.library == "corenlp":
            doc = self.nlp.annotate(text)
            pos_tags = [token.pos for sent in doc.sentence for token in sent.token]
        return pos_tags

    def cleanup(self):
        if self.library == "corenlp" and isinstance(self.nlp, CoreNLPClient):
            self.nlp.stop()


####################################
# Base Evaluators
####################################


class BaseEvaluator(ABC):
    @abstractmethod
    def __call__(self, texts: List[str]) -> Dict[str, float]:
        pass

    def cleanup(self):
        pass


class NLPEvaluator(BaseEvaluator):
    def __init__(
        self,
        language: str = "en",
        library: Literal["spacy", "stanza", "corenlp"] = "spacy",
    ):
        self.language = language
        self.nlp = NLProcessor(language, library)

    def cleanup(self):
        self.nlp.cleanup()


class ClassifierEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_name_or_path: str,
        label: str,
        out_key: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.label = label
        self.out_key = out_key
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            truncation=True,
            batch_size=8,
            max_length=512,
        )

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        scores = []
        out_key = self.out_key or self.label
        for text in texts:
            score = self.classifier(text, top_k=self.model.config.num_labels)
            scores.append(extract_score(score, self.label))
        return {out_key: np.mean(scores)}

    def cleanup(self):
        del self.classifier
        del self.tokenizer
        del self.model


####################################
# Evaluation Clases
####################################


class SentenceLenghtEvaluator(NLPEvaluator):

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        sent_lengths = []
        for text in texts:
            sents = self.nlp.sents(text)
            for sent in sents:
                sent_lengths.append(len(sent.split()))
        return {"length_more": np.mean(sent_lengths)}


class FunctionWordEvaluator(NLPEvaluator):

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        function_word_count = 0
        total_word_count = 0
        for text in texts:
            pos_tags = self.nlp.pos_tags(text)
            function_word_count += sum(
                1 for tag in pos_tags if tag in self.nlp.function_word_pos_tags
            )
            total_word_count += sum(1 for token in text if token.isalpha())

        if total_word_count == 0:
            ratio = 0.0
        else:
            ratio = function_word_count / total_word_count

        return {"function_more": ratio}


class GradeLevelEvaluator(NLPEvaluator):

    def __init__(
        self,
        language: str = "en",
        library: Literal["spacy", "stanza", "corenlp"] = "spacy",
    ):
        super().__init__(language, library)
        self.readability = None
        if language == "zh":
            self.readability = ChineseReadability(nlp_provider=JiebaNLP())

    def get_grade_level(self, text: str) -> float:
        if self.nlp.language == "en":
            fk_text = textstat.flesch_kincaid_grade(text)
            lw_text = textstat.linsear_write_formula(text)
            gf_text = textstat.gunning_fog(text)
            return np.nanmean([fk_text, lw_text, gf_text])
        elif self.nlp.language == "zh":
            sentences = [s.strip() for s in self.readability.stnsplit.split(text) if s.strip()]
            return self.readability.chengyong_gf0025_readability(sentences)
        else:
            return textstat.flesch_reading_ease(text)

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        averages = []
        for text in texts:
            averages.append(self.get_grade_level(text))
        return {"grade_more": np.nanmean(averages)}


class SentenceClassifierEvaluator(ClassifierEvaluator, NLPEvaluator):

    def __init__(
        self,
        model_name_or_path: str,
        label: str,
        language: str = "en",
        library: Literal["spacy", "stanza", "corenlp"] = "spacy",
        out_key: Optional[str] = None,
    ):
        ClassifierEvaluator.__init__(self, model_name_or_path, label, out_key)
        NLPEvaluator.__init__(self, language, library)

    def score(self, text: str) -> float:
        sents = self.nlp.sents(text)
        sentences = [sent.text.strip() for sent in sents]
        scores = [
            ClassifierEvaluator.__call__(self, [sent])[self.label] for sent in sentences
        ]
        return np.mean(scores).item() if len(sentences) > 0 else 0

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        scores = []
        out_key = self.out_key or self.label
        for text in texts:
            scores.append(self.score(text))
        return {out_key: np.mean(scores)}


class MultiClassEvaluator(ClassifierEvaluator):

    def __init__(
        self,
        model_name_or_path: str,
        label: str,
        out_key: Optional[str] = None,
    ):
        ClassifierEvaluator.__init__(self, model_name_or_path, label, out_key)

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        scores = defaultdict(list)
        for result in self.classifier(texts, top_k=1):
            scores[result[0]["label"]].append(result[0]["score"])

        all_labels = self.model.config.id2label.values()
        return {label: np.mean(scores.get(label, [0])) for label in all_labels}


####################################
# Evaluation Runner
####################################


@dataclass
class EvaluationRunner:
    evaluators: List[BaseEvaluator]
    lang: str
    library: str

    def __init__(
        self,
        classifier_config: Dict[str, str],
        lang: str = "en",
        library: str = "spacy",
    ):
        self.lang = lang
        self.library = library
        self.evaluators = [
            SentenceLenghtEvaluator(lang, library),
            FunctionWordEvaluator(lang, library),
            GradeLevelEvaluator(lang, library),
        ]

        for axis in classifier_config.keys():
            if classifier_config[axis]["label"] == "*":
                self.evaluators.append(MultiClassEvaluator(**classifier_config[axis]))
            else:
                self.evaluators.append(ClassifierEvaluator(**classifier_config[axis]))

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        results = {}
        for evaluator in self.evaluators:
            results.update(evaluator(texts))
        return results

    def cleanup(self):
        for evaluator in self.evaluators:
            evaluator.cleanup()
        torch.cuda.empty_cache()
