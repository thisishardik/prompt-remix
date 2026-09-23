import os
import sys
import re
import math
import random
import logging
from argparse import Namespace
from typing import List, Tuple

import numpy as np
import torch
import spacy
from transformers import (
    AutoModelForCausalLM,
    GPT2Tokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GenerationConfig,
)

# Add JAMDecoding to sys.path ahead of hiatus so `import src.*` resolves to
# JAMDecoding/src (hiatus also ships a top-level `src` package).
JAMDEC_ROOT = "/mmfs1/home/hardiksr/repository/JAMDecoding"
for _key in list(sys.modules):
    if _key == "src" or _key.startswith("src."):
        del sys.modules[_key]
if JAMDEC_ROOT in sys.path:
    sys.path.remove(JAMDEC_ROOT)
sys.path.insert(0, JAMDEC_ROOT)

from neurologic_super_fast.generate import GenerationWrapper
from neurologic_super_fast.util import NeuroLogicConfig
from keyword_extraction_methods.likelihood_extractors import (
    get_likelihood_constraints_gpt2,
    get_likelihood_constraints_infill,
)
from keyword_extraction_methods.auto_extractors import get_keybert_constraints
from src.utils import (
    prepare_constraints,
    pre_process_constraints,
    combine_constraints,
    create_complete_sentence,
    calc_cola_score,
)
from src.medium_constraint import get_synon_words, get_lemmatized_words, remove_repeated_tokens
from src.filter import SummFilter


class JAMDECGeneration:
    """End-to-end JAMDEC obfuscation pipeline."""

    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = getattr(args, "cache_dir", None)
        fast = getattr(args, "fast", False) or os.environ.get("JAMDEC_FAST", "").lower() in (
            "1",
            "true",
            "yes",
        )

        if fast:
            self.jamdec_args = Namespace(
                device=self.device,
                cache_dir=cache_dir,
                keyword_extractors=["likelihood-gpt2"],
                likelihood_p_threshold=0.5,
                p_content_words=0.8,
                keybert_length=2,
                batch_size=4,
                top_k_medium_constraints=4,
                like_words=False,
                similar_words=False,
                generation_batch_size=2,
                max_new_tokens=2,
                max_decode_tokens=384,
                num_beams=5,
                num_return_sequences=5,
                no_repeat_ngram=3,
                do_sample=[False],
                min_length=2,
                repetition_penalty=1.0,
                min_length_to_generate=3,
                ordered=[True],
                grouping_strategy="type",
                likelihood_prune_factor=0.4,
                constraint_prune_factor=0.6,
                diversity=[False],
                do_early_stopping=True,
                constraint_set_names=["constraints_only"],
                max_candidates_before_filter=24,
                nli_threshold=0.8,
                nli_k=5,
                nli_batch_size=5,
                cola_threshold=0.0,
                cola_k=3,
                eval_nli_threshold=0.8,
                eval_cola_threshold=0.8,
            )
            logging.info("JAMDEC fast mode enabled (likelihood-gpt2, constraints_only, beam=5).")
        else:
            self.jamdec_args = Namespace(
                device=self.device,
                cache_dir=cache_dir,
                # Keyword extraction
                keyword_extractors=["likelihood-gpt2", "likelihood-infill", "keybert"],
                likelihood_p_threshold=0.5,
                p_content_words=0.8,
                keybert_length=2,
                batch_size=4,
                # Medium constraints
                top_k_medium_constraints=4,
                like_words=True,
                similar_words=True,
                # Generation
                generation_batch_size=2,
                max_new_tokens=2,
                max_decode_tokens=512,
                num_beams=10,
                num_return_sequences=10,
                no_repeat_ngram=3,
                do_sample=[True, False],
                min_length=2,
                repetition_penalty=1.0,
                min_length_to_generate=3,
                # Neurologic
                ordered=[True, False],
                grouping_strategy="type",
                likelihood_prune_factor=0.4,
                constraint_prune_factor=0.6,
                diversity=[True, False],
                do_early_stopping=True,
                constraint_set_names=[
                    "constraints_like_similar",
                    "constraints_like",
                    "constraints_only",
                ],
                max_candidates_before_filter=None,
                # Filtering
                nli_threshold=0.8,
                nli_k=5,
                nli_batch_size=5,
                cola_threshold=0.0,
                cola_k=3,
                # Evaluation selection
                eval_nli_threshold=0.8,
                eval_cola_threshold=0.8,
            )

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        logging.info("Loading GPT2-XL...")
        self.tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2-xl", cache_dir=cache_dir)
        self.tokenizer_gpt2.pad_token = self.tokenizer_gpt2.eos_token
        self.tokenizer_gpt2.padding_side = "left"
        self.model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2-xl", cache_dir=cache_dir).to(self.device)
        self.model_gpt2.config.pad_token_id = self.model_gpt2.config.eos_token_id

        bad_token = ['\n', ':', "'", '-', '_', '@', 'Ċ', 'Ġ:']
        self.bad_words_ids = []
        for t in bad_token:
            ids = self.tokenizer_gpt2.convert_tokens_to_ids([t])
            if isinstance(ids, list) and ids and all(isinstance(i, int) and i >= 0 for i in ids):
                self.bad_words_ids.append(ids)
        if not self.bad_words_ids:
            self.bad_words_ids = None

        self._gen_wrapper = GenerationWrapper(self.model_gpt2)

        extractors = set(self.jamdec_args.keyword_extractors)
        if "likelihood-infill" in extractors:
            logging.info("Loading T5-base...")
            self.model_t5 = T5ForConditionalGeneration.from_pretrained(
                "t5-base", cache_dir=cache_dir
            ).to(self.device)
            self.tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir=cache_dir)
        else:
            self.model_t5 = None
            self.tokenizer_t5 = None

        logging.info("Loading NLI model...")
        nli_model_name = "alisawuffles/roberta-large-wanli"
        self.jamdec_args.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name, cache_dir=cache_dir)
        self.jamdec_args.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name, cache_dir=cache_dir
        ).to(self.device)

        logging.info("Loading CoLA model...")
        self.jamdec_args.cola_model = AutoModelForSequenceClassification.from_pretrained(
            "textattack/roberta-base-CoLA", cache_dir=cache_dir
        ).to(self.device)
        self.jamdec_args.cola_tokenizer = AutoTokenizer.from_pretrained(
            "textattack/roberta-base-CoLA", cache_dir=cache_dir
        )

        logging.info("JAMDEC models loaded.")

    def _split_into_sentences(self, text: str) -> Tuple[List[str], List[str]]:
        """Split text into y_orig and x_l (left context) pairs."""
        import re
        sentences = re.split(r'(?<=[.!?。！？\n])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [text.strip()]
        
        chunked_sentences = []
        for s in sentences:
            token_count = len(self.tokenizer_gpt2(s)["input_ids"])
            if token_count <= 200:
                chunked_sentences.append(s)
            else:
                words = s.split()
                current_chunk = []
                current_tokens = 0
                for w in words:
                    w_tokens = len(self.tokenizer_gpt2(w)["input_ids"])
                    if current_tokens + w_tokens > 200 and current_chunk:
                        chunked_sentences.append(" ".join(current_chunk))
                        current_chunk = [w]
                        current_tokens = w_tokens
                    else:
                        current_chunk.append(w)
                        current_tokens += w_tokens
                if current_chunk:
                    chunked_sentences.append(" ".join(current_chunk))

        y_orig_ls = []
        x_l_ls = []

        for i, sent in enumerate(chunked_sentences):
            y_orig_ls.append(sent)
            if i == 0:
                x_l_ls.append(sent)  # first sentence uses itself as context
            else:
                x_l_ls.append(chunked_sentences[i - 1])  # use previous sentence as context

        return y_orig_ls, x_l_ls

    def _extract_keywords(self, y_orig: str, prefix: str, save_idx: int,
                          likelihood_constraints_gpt2_ls: list) -> dict:
        """Extract keywords using all configured methods and prepare constraints."""
        constraint_ls = {}
        args = self.jamdec_args

        for keyword_extractor in args.keyword_extractors:
            try:
                if keyword_extractor == "likelihood-infill":
                    keyword_list = get_likelihood_constraints_infill(
                        y_orig, self.model_t5, self.tokenizer_t5, args
                    )
                elif keyword_extractor == "likelihood-gpt2":
                    keyword_list = likelihood_constraints_gpt2_ls[save_idx]
                elif keyword_extractor == "keybert":
                    keyword_list = get_keybert_constraints(y_orig, args)
                else:
                    continue
            except Exception as e:
                logging.warning(
                    "Keyword extractor %s failed: %s", keyword_extractor, e
                )
                continue

            # Order keywords by position in original text
            y_orig_lower = y_orig.lower()
            ordered_keyword_list = []
            for keyword in keyword_list:
                if not keyword:
                    continue
                keyword_start_idx = y_orig_lower.find(keyword.lower())
                if keyword_start_idx < 0:
                    continue
                ordered_keyword_list.append((
                    y_orig[keyword_start_idx:keyword_start_idx + len(keyword)],
                    keyword_start_idx,
                ))
            ordered_keyword_list = sorted(ordered_keyword_list, key=lambda x: x[-1])
            constraint_words = [o[0] for o in ordered_keyword_list]

            # Pre-process
            constraint_words = pre_process_constraints(constraint_words, y_orig, prefix)
            i = 0
            while constraint_words == []:
                keyword_list = random.sample(
                    [y for y in y_orig.split(" ") if y != ""], 1
                )
                constraint_words = pre_process_constraints(keyword_list, y_orig, prefix)
                i += 1
                if i > 100:
                    break

            if not constraint_words:
                continue

            similar_ls = [[]] * len(constraint_words)
            like_ls = [[]] * len(constraint_words)
            if args.similar_words:
                similar_ls = get_synon_words(constraint_words, y_orig)
            if args.like_words:
                like_ls = get_lemmatized_words(constraint_words)

            constraint_list_like = combine_constraints(constraint_words, like_ls)
            constraint_list_like_similar = combine_constraints(
                constraint_words, [s + l for s, l in zip(similar_ls, like_ls)]
            )

            constraint_list_like = remove_repeated_tokens(constraint_list_like, self.tokenizer_gpt2)
            constraint_list_like_similar = remove_repeated_tokens(
                constraint_list_like_similar, self.tokenizer_gpt2
            )

            all_variants = {
                "constraints_like_similar": [
                    prepare_constraints(self.tokenizer_gpt2, constraint_list_like_similar)
                ],
                "constraints_like": [
                    prepare_constraints(self.tokenizer_gpt2, constraint_list_like)
                ],
                "constraints_only": [
                    prepare_constraints(self.tokenizer_gpt2, constraint_words)
                ],
            }
            selected = getattr(args, "constraint_set_names", list(all_variants.keys()))
            constraint_ls[keyword_extractor] = {
                name: all_variants[name]
                for name in selected
                if name in all_variants
            }

        return constraint_ls

    def _generate_candidates(self, y_orig: str, prefix: str, keywords: dict) -> List[str]:
        """Over-generate candidate obfuscations using CoDi-BS."""
        args = self.jamdec_args
        all_generations = []

        input_encoding = self.tokenizer_gpt2([prefix], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        prefix_len = input_encoding["input_ids"].shape[1]
        y_orig_token_len = len(self.tokenizer_gpt2(y_orig)["input_ids"])
        max_decode = getattr(args, "max_decode_tokens", 512)
        max_new_length = min(
            int(y_orig_token_len * args.max_new_tokens),
            1024 - prefix_len,
            max_decode,
        )
        if max_new_length <= 0:
            return []

        candidate_cap = getattr(args, "max_candidates_before_filter", None)

        for keywords_name in keywords.keys():
            for constraints_ls_name in keywords[keywords_name].keys():
                constraints = keywords[keywords_name][constraints_ls_name][0]
                if not constraints:
                    continue

                for do_sample in args.do_sample:
                    generation_config = GenerationConfig(
                        max_new_tokens=max_new_length,
                        num_return_sequences=args.num_return_sequences,
                        pad_token_id=self.model_gpt2.config.pad_token_id,
                        num_beams=args.num_beams,
                        no_repeat_ngram_size=args.no_repeat_ngram,
                        do_sample=do_sample,
                    )

                    for ordered in args.ordered:
                        for diversity in args.diversity:
                            neurologic_config = NeuroLogicConfig(
                                constraints=[constraints],
                                ordered=ordered,
                                grouping_strategy=args.grouping_strategy,
                                likelihood_prune_factor=args.likelihood_prune_factor,
                                constraint_prune_factor=args.constraint_prune_factor,
                                constraint_prune_number=int(
                                    len(constraints) * args.constraint_prune_factor
                                ),
                                diversity=diversity,
                                do_early_stopping=args.do_early_stopping,
                            )

                            try:
                                outputs = self._gen_wrapper.generate(
                                    **input_encoding,
                                    generation_config=generation_config,
                                    neurologic_config=neurologic_config,
                                    no_repeat_ngram_size=args.no_repeat_ngram,
                                    bad_words_ids=self.bad_words_ids,
                                    min_length=args.min_length,
                                    repetition_penalty=args.repetition_penalty,
                                )
                                decoded = self.tokenizer_gpt2.batch_decode(
                                    outputs, skip_special_tokens=True
                                )
                                generations = [g[len(prefix):] for g in decoded]
                                all_generations.extend(generations)
                            except Exception as e:
                                logging.warning("Generation failed for a config: %s", e)
                                # CUDA device-side asserts poison the context; clear so later
                                # configs/sentences can still run instead of cascading failures.
                                if torch.cuda.is_available():
                                    try:
                                        torch.cuda.synchronize()
                                    except Exception:
                                        pass
                                    try:
                                        torch.cuda.empty_cache()
                                    except Exception:
                                        pass
                                continue

                            if (
                                candidate_cap is not None
                                and len(all_generations) >= candidate_cap
                            ):
                                return all_generations

        return all_generations

    def _filter_candidates(self, y_orig: str, candidates: List[str]) -> str:
        """Filter candidates using NLI + CoLA and return the best one."""
        args = self.jamdec_args

        candidates = create_complete_sentence(candidates)
        candidates = list(dict.fromkeys(candidates))  # deduplicate preserving order
        candidates = [g for g in candidates if len(g) > 1]

        if not candidates:
            return y_orig

        generations_list = [[g] for g in candidates]
        y_orig_list = [y_orig] * len(generations_list)

        summ_filter = SummFilter(args)
        try:
            out_sample_topk, out_cola_topk, out_nli_topk = summ_filter.filter_all(
                y_orig_list, generations_list,
                args.nli_threshold, args.nli_k, args.cola_k, "topk"
            )
        except Exception as e:
            logging.warning("Filtering failed: %s", e)
            return y_orig
        finally:
            del summ_filter
            torch.cuda.empty_cache()

        if not out_sample_topk:
            return y_orig

        cola_options = []
        for i, nli_val in enumerate(out_nli_topk):
            if nli_val > args.eval_nli_threshold:
                cola_options.append(out_cola_topk[i])
            else:
                cola_options.append(0)

        for i in range(len(cola_options)):
            if cola_options[i] < args.eval_cola_threshold:
                cola_options[i] = 0

        if len(cola_options) > 0 and max(cola_options) > 0:
            best_idx = cola_options.index(max(cola_options))
            return out_sample_topk[best_idx]["pair"][0]

        return y_orig

    def generate(self, prompt: str):
        """Run the full JAMDEC pipeline on a document.

        Args:
            prompt: The full text to obfuscate.

        Returns:
            Tuple of ([obfuscated_text], metadata_dict, score)
        """
        args = self.jamdec_args

        y_orig_ls, x_l_ls = self._split_into_sentences(prompt)

        if not y_orig_ls:
            return [prompt], {}, 0.0

        from torch.utils.data import DataLoader
        from src.utils import Dataset

        dataset = Dataset(y_orig_ls, x_l_ls)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        likelihood_constraints_gpt2_ls = []
        if "likelihood-gpt2" in args.keyword_extractors:
            for y_orig_batch, prefix_batch in dataloader:
                keyword_list = get_likelihood_constraints_gpt2(
                    self.tokenizer_gpt2, self.model_gpt2,
                    y_orig_batch, prefix_batch, args
                )
                likelihood_constraints_gpt2_ls.extend(keyword_list)

        obfuscated_sentences = []
        metadata = []

        for idx, (y_orig, prefix) in enumerate(zip(y_orig_ls, x_l_ls)):
            if len(y_orig.split(" ")) <= args.min_length_to_generate:
                obfuscated_sentences.append(y_orig)
                metadata.append({"sentence": y_orig, "status": "too_short"})
                continue

            keywords = self._extract_keywords(
                y_orig, prefix, idx, likelihood_constraints_gpt2_ls
            )

            if not keywords:
                obfuscated_sentences.append(y_orig)
                metadata.append({"sentence": y_orig, "status": "no_keywords"})
                continue

            candidates = self._generate_candidates(y_orig, prefix, keywords)

            if not candidates:
                obfuscated_sentences.append(y_orig)
                metadata.append({"sentence": y_orig, "status": "no_candidates"})
                continue

            best = self._filter_candidates(y_orig, candidates)
            obfuscated_sentences.append(best)
            metadata.append({
                "sentence": y_orig,
                "status": "obfuscated" if best != y_orig else "fallback_original",
                "num_candidates": len(candidates),
            })

        final_generation = " ".join(obfuscated_sentences)

        return [final_generation], {"sentences": metadata}, 1.0
