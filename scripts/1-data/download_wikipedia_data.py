import argparse
import itertools
import json
import os
import random
import string

from concurrent.futures import ProcessPoolExecutor, as_completed

import jieba
import numpy as np
import pycld2 as cld2
import spacy
from datasets import load_dataset
from stanza.server import CoreNLPClient
from tqdm import tqdm

# Global variables for multiprocessing
GLOBAL_DATASET = None
PARAGRAPH_FILTER = None
CORENLP_CLIENT = None  # Only initialized if needed
CORENLP_LANGUAGES = {
    "zh": "chinese",
    "ar": "arabic",
    "en": "english",
}

def initialize_worker(dataset_name: str, language: str, date: str, seed: int) -> None:
    """
    Initializes global variables for each worker process.
    This function is called once per worker process.
    """
    global GLOBAL_DATASET
    global PARAGRAPH_FILTER

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Initialize ParagraphFilter
    PARAGRAPH_FILTER = ParagraphFilter(lang=language)

    # Load the dataset without shuffling
    print(f"Process {os.getpid()} loading dataset...")
    dataset = load_dataset(
        dataset_name,
        language=language,
        date=date,
        split="train[:100000]",
        # num_proc=1  # Ensuring single process per worker to avoid nested parallelism
        num_proc=1,  # Load dataset in each worker process
        trust_remote_code=True
    )

    GLOBAL_DATASET = dataset
    print(f"Process {os.getpid()} finished loading dataset.")


class ParagraphFilter:
    # Modified here (max_wc, min_wc)
    def __init__(
        self,
        max_wc: int = 150,
        min_wc: int = 100,
        max_para: int = 20,
        min_para: int = 2,
        lang: str = "zh",
        lang_threshold: float = 0.5,
    ) -> None:
        """
        Create a ParagraphFilter to validate and trim paragraphs.
        """
        global CORENLP_CLIENT
        self.MAX_WC = max_wc
        self.MIN_WC = min_wc
        self.MAX_PARA = max_para
        self.MIN_PARA = min_para
        self.lang = lang
        self.lang_threshold = lang_threshold

        if self.lang == "ru":
            self.nlp = spacy.load("ru_core_news_lg")
        elif self.lang == "en":
            self.nlp = spacy.load("en_core_web_lg")
        elif self.lang == "zh":
            self.nlp = spacy.load("zh_core_web_lg")
        elif self.lang == "ar":
            global CORENLP_CLIENT
            if CORENLP_CLIENT is None:
                CORENLP_CLIENT = CoreNLPClient(
                    endpoint="http://localhost:8000",
                    annotators=["tokenize", "ssplit", "pos", "lemma"],
                    timeout=30000,
                    memory="6G",
                    be_quiet=True,
                    start_server=True,
                    properties=CORENLP_LANGUAGES[self.lang]
                )
                CORENLP_CLIENT.ensure_alive()
            self.nlp = CORENLP_CLIENT
        else:
            raise ValueError(f"Unsupported language: {self.lang}")

    def is_valid_language(self, text: str) -> bool:
        """
        Detect whether `text` is in the filter's target language with sufficient confidence.
        """
        try:
            is_reliable, text_bytes, top_languages = cld2.detect(text, bestEffort=False)
            languages = {
                language_code: percent
                for (language_name, language_code, percent, score) in top_languages
            }
            language_present = False
            language_confident = False
            if self.lang == "zh":
                language_present = is_reliable and (
                    "zh" in languages or "zh-Hant" in languages
                )
                language_confident = (
                    max(languages.get("zh", 0), languages.get("zh-Hant", 0)) / 100.0
                    > self.lang_threshold
                )
            else:
                language_present = is_reliable and (self.lang in languages)
                language_confident = (
                    languages.get(self.lang, 0) / 100.0 > self.lang_threshold
                )
            return language_present and language_confident
        except Exception:
            return False

    def word_count(self, text: str) -> int:
        """
        Return an approximate word/token count for `text`.
        """
        if self.lang == "zh":
            return len(jieba.lcut(text))
        else:
            return len(text.translate(str.maketrans("", "", string.punctuation)).split())

    def valid_paragraph(self, paragraph: str) -> str:
        """
        Validate and (optionally) trim a paragraph to meet size and language rules.
        """
        if not self.is_valid_language(paragraph):
            return None

        sentences = []

        if self.lang == "ar":
            sents = self.nlp.annotate(paragraph).sentence
            for sentence in sents:
                sent_text = " ".join([token.word for token in sentence.token])
                sentences.append(sent_text.strip())
        else:
            sentences = [sent.text.strip() for sent in self.nlp(paragraph).sents]

        if len(sentences) < self.MIN_PARA:
            return None

        sent_word_counts = [self.word_count(sent) for sent in sentences]

        current_para = []
        curr_length = 0
        for sent, wc in zip(sentences, sent_word_counts):
            if curr_length + wc <= self.MAX_WC:
                current_para.append(sent)
                curr_length += wc
            else:
                break

        if curr_length < self.MIN_WC:
            return None

        return " ".join(current_para)

    def valid_paragraphs(self, paragraphs: list) -> list:
        """
        Apply `valid_paragraph` to a list of paragraphs and return the valid ones.
        """
        valid_paragraphs = []
        for para in paragraphs:
            para = self.valid_paragraph(para)
            if para:
                valid_paragraphs.append(para)
        return valid_paragraphs


def process_entry(idx: int, end_markers: list) -> dict:
    """
    Processes a single dataset entry to extract a valid paragraph.
    This function must be a top-level function to be picklable for ProcessPoolExecutor.
    """
    global GLOBAL_DATASET
    global PARAGRAPH_FILTER

    entry = GLOBAL_DATASET[idx]
    text = entry["text"]

    # Remove sections after specified markers
    for marker in end_markers:
        if marker in text:
            text = text.split(marker)[0]
            break  # Assume only one marker is present

    # Split into paragraphs
    paragraphs = text.split("\n\n")
    paragraphs = list(itertools.chain(*[p.split("\n") for p in paragraphs]))

    # Filter valid paragraphs
    valid_paras = PARAGRAPH_FILTER.valid_paragraphs(paragraphs)
    # valid_paras = [para.strip() for para in paragraphs if PARAGRAPH_FILTER.valid_paragraph(para)]
    if not valid_paras:
        return None

    # Randomly select one paragraph
    selected_para = random.choice(valid_paras)
    return {"id": "Wikipedia_" + entry.get("title", ""), "original": selected_para}


class DataExtractor:
    def __init__(
        self,
        dataset_name: str,
        language: str,
        date: str,
        n_samples: int,
        output_path: str,
        num_workers: int = 4,
        seed: int = 0,
    ) -> None:
        """
        Object for extracting paragraphs from a dataset.
        """
        self.dataset_name = dataset_name
        self.language = language
        self.date = date
        self.n_samples = n_samples
        self.output_path = output_path
        self.num_workers = num_workers
        self.seed = seed

        # Initialize random seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.paragraph_filter = ParagraphFilter()
        self.save_dir = os.path.dirname(self.output_path)
        os.makedirs(self.save_dir, exist_ok=True)

        # End markers to truncate text
        self.end_markers = [
            "Примечания\n",
            "Примечания \n",
            "Ссылки\n",
            "Ссылки \n",
            "Литература\n",
            "Литература \n",
            "See also\n",
            "See also \n",
            "References\n",
            "References \n",
            "Bibliography\n",
            "Bibliography \n",
            "参见\n",
            "参见 \n",
            "外部链接\n",
            "外部链接 \n",
            "外部連結\n",
            "外部連結 \n"  # See also, references, external links,
            "انظر أيضًا\n",
            "انظر أيضًا \n",  # See also
            "المراجع\n",
            "المراجع \n",  # References
            "المصادر\n",
            "المصادر \n",  # Sources / Bibliography
            "مراجع\n",
            "مراجع \n",  # Alternative for References
            "وصلات خارجية\n",
            "وصلات خارجية \n",  # External links
            "ملاحظات\n",
            "ملاحظات \n",  # Notes
        ]

    def load_data_length(self):
        """
        Loads the dataset length for processing.
        Loading the entire dataset in each process is handled in the worker initializer.
        """
        # To get the length, load the dataset once in the main process
        dataset = load_dataset(
            self.dataset_name,
            language=self.language,
            date=self.date,
            split="train[:100000]",  #
            # num_proc=1  # Ensuring single process
            num_proc=self.num_workers,  # Load dataset in each worker process
            trust_remote_code=True
        )
        total_length = len(dataset)
        del dataset  # Free memory
        return total_length

    def extract(self):
        """
        Run the extraction pipeline across worker processes and collect samples.
        """
        print("Starting data extraction...")

        # Get the total number of samples in the dataset
        total_length = self.load_data_length()
        print(f"Total samples in dataset: {total_length}")

        wiki_train_data = []
        count = 0

        # Initialize ProcessPoolExecutor with initializer
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=initialize_worker,
            initargs=(self.dataset_name, self.language, self.date, self.seed),
        ) as executor:
            # Generate a list of indices to process
            indices = list(range(5000, total_length))
            # No shuffling to maintain original order

            # Submit tasks to the executor
            futures = {
                executor.submit(process_entry, idx, self.end_markers): idx
                for idx in indices
            }

            # Use tqdm to display progress
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing entries"
            ):
                try:
                    result = future.result()
                    if result:
                        wiki_train_data.append(result)
                        count += 1
                        if count >= self.n_samples:
                            print(
                                f"Desired sample count {self.n_samples} reached. Cancelling remaining tasks..."
                            )
                            break
                    if count % 100 == 0 and count > 0:
                        print(f"Sampled: {count}")
                except Exception as e:
                    print(f"Error processing a future: {e}")

            # After reaching desired count, cancel all pending futures
            pending_futures = [f for f in futures if not f.done()]
            for f in pending_futures:
                cancelled = f.cancel()
                if cancelled:
                    print(f"Cancelled future for index {futures[f]}")
                else:
                    print(f"Could not cancel future for index {futures[f]}")

        print(f"Total sampled paragraphs: {count}")
        return wiki_train_data

    def save_data(self, data: list) -> None:
        """
        Save extracted data to the configured output path in JSON format.
        """
        global CORENLP_CLIENT
        # Only stop CoreNLP server if it was started (i.e., for Arabic)
        if CORENLP_CLIENT is not None:
            CORENLP_CLIENT.stop()
            print("Stopping CoreNLP server...")
        print(f"Saving data to {self.output_path}...")
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("Data saved successfully.")

    def run(self):
        """
        Convenience method to run extraction and persist results.
        """
        extracted_data = self.extract()
        self.save_data(extracted_data)


def parse_arguments():
    """
    Parse command line arguments for running the extraction script.
    """
    parser = argparse.ArgumentParser(description="Extract paragraphs from a dataset.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikipedia",
        help="Name of the dataset to load.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",  # Modify here.
        help="Language of the dataset.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="20250201",  # Modify here.
        help="Date of the dataset snapshot.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,  # Modify here.
        help="Number of paragraphs to sample.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/gscratch/stf/mpotto/hiatus/data/zh/raw/wiki.jsonl",  # Modify here.
        help="Path to save the extracted data.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for processing.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    return parser.parse_args()


def main():
    """
    Parse arguments, construct a DataExtractor, and run it.
    """
    args = parse_arguments()
    extractor = DataExtractor(
        dataset_name=args.dataset_name,
        language=args.language,
        date=args.date,
        n_samples=args.n_samples,
        output_path=args.output_path,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    extractor.run()


if __name__ == "__main__":
    main()

# python scripts/1-data/download_wiki_data.py --language=ar --date=20231101 --n_samples=50000 --num_workers=4 --output_path=/home/hardiksr/assets/hiatus/data/ar/raw/data_ar.json
