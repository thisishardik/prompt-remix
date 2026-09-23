import argparse
import json
import os
import pprint
import gc
import torch
from typing import Dict, List, Optional, Union, Tuple

import pydantic
import pandas as pd
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from tqdm import tqdm

from hiatus.evaluation.huggingface_pipeline import HuggingFaceLLMAdapter
from hiatus.evaluation.io_utils import fopen
from hiatus.evaluation.logger import logger
from hiatus.evaluation.meaning_sim_contextual_embeddings import (
    MultiLingualDocumentMeaningSimilarityMetric,
)

class SenseScores(pydantic.BaseModel):
    """
    A Pydantic model to parse and validate integer scores (0 to 20 inclusive)
    for each aspect of the evaluation.
    """

    semantic_coverage: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for how well the privatized text covers the semantic meaning of the original text.",
    )
    factuality: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for whether the privatized text preserves factual statements from the original text.",
    )
    informativeness: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for whether the privatized text maintains key informational content.",
    )
    relevance: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for whether the privatized text is relevant to the original text's topic.",
    )
    specificity: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for how well the privatized text preserves specific details of the original text.",
    )
    correctness: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for whether the privatized text avoids introducing factual/logical errors.",
    )
    accuracy: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for how accurately the privatized text reflects the original text details.",
    )
    semantically_appropriate: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for whether the privatized text remains semantically aligned to the original.",
    )
    consistency: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for whether the privatized text is self-consistent and consistent with the original.",
    )
    coherence: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for the logical structure and flow of the privatized text.",
    )
    fluency: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for the overall fluency and readability of the privatized text.",
    )
    quality: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for the overall quality of the privatized text.",
    )
    understandability: Optional[int] = pydantic.Field(
        None,
        ge=0,
        le=20,
        description="Score for how easy it is to understand the privatized text compared to the original.",
    )

# Example expanded default criteria
DEFAULT_CRITERIA = {
    "semantic_coverage": "Does text2 preserve the overall semantic meaning from text1?",
    "factuality": "Does text2 preserve the factual statements of text1 accurately?",
    "informativeness": "Does text2 maintain the key informational content of text1?",
    "relevance": "Does text2 remain relevant to the topic and context of text1?",
    "specificity": "Does text2 preserve specific details found in text1?",
    "correctness": "Is text2 free from any newly introduced errors or inconsistencies not present in text1?",
    "accuracy": "Does text2 accurately reflect the details from text1 without misrepresentation?",
    "semantically_appropriate": "Is text2 aligned semantically with the original meaning and intent of text1?",
    "consistency": "Is text2 logically and topically consistent with itself and with text1?",
    "coherence": "Is text2 coherent and well-structured, retaining the clarity from text1?",
    "fluency": "Is text2 written fluently and naturally, without awkward phrasing?",
    "quality": "Overall, how would you rate the quality of text2 compared to text1?",
    "understandability": "Is text2 as understandable as text1 while preserving its key points?",
}

DEFAULT_CRITERIA = {
    "semantic_coverage": (
        "When comparing text2 to text1, assess how thoroughly text2 covers the "
        "essential semantic elements, key ideas, and overall message found in text1. "
        "A higher score indicates text2 captures the comprehensive meaning, themes, "
        "and subtext from text1 without omitting or significantly altering them."
    ),
    "factuality": (
        "Check whether the statements or claims in text2 accurately reflect any "
        "factual information in text1. A high score means text2 preserves true "
        "statements from text1 and does not introduce new factual errors, "
        "misinterpretations, or contradictions."
    ),
    "informativeness": (
        "Determine whether text2 maintains the key informational content from text1. "
        "A high score indicates text2 includes the important details, data, and context "
        "presented in text1 without significant omissions."
    ),
    "relevance": (
        "Evaluate whether text2 remains centered on the same topic or purpose as text1. "
        "A high score means text2 does not wander off topic and continues to focus on "
        "the essential subject matter introduced in text1."
    ),
    "specificity": (
        "Assess whether text2 preserves specific details found in text1 (such as names, "
        "figures, dates, locations, or other unique attributes). A high score indicates "
        "text2 retains these details accurately rather than generalizing or omitting them."
    ),
    "correctness": (
        "Check if text2 introduces any new errors or inconsistencies that were not present "
        "in text1. A high score means text2 avoids adding incorrect information, "
        "logical contradictions, or factual mistakes beyond what was in text1."
    ),
    "accuracy": (
        "Determine how accurately text2 reflects the details and essence of text1 without "
        "misrepresentation. A high score means text2 stays faithful to text1's facts, "
        "claims, and nuances."
    ),
    "semantically_appropriate": (
        "Assess whether text2 remains aligned with text1 in terms of meaning, tone, and intent. "
        "A high score indicates text2 preserves the overall purpose and message of text1 without "
        "deviating into unintended interpretations."
    ),
    "consistency": (
        "Evaluate if text2 is consistent in its own statements and does not introduce "
        "contradictions relative to text1. A high score indicates text2's content is logically "
        "and thematically consistent with itself and with text1."
    ),
    "coherence": (
        "Assess if text2 is well-organized, logically structured, and easy to follow. "
        "A high score indicates the flow of text2 is smooth, and the ideas connect naturally, "
        "similar to how they appear in text1."
    ),
    "fluency": (
        "Examine the linguistic quality, grammar, and readability of text2. A high score "
        "means text2 is written in a clear, natural, and error-free manner, ensuring it is "
        "pleasant to read."
    ),
    "quality": (
        "Provide an overall, holistic assessment of text2 compared to text1. Consider "
        "clarity, style, depth, and alignment with text1's intentions. A higher score means "
        "text2 meets or exceeds expectations of completeness, clarity, and correctness."
    ),
    "understandability": (
        "Determine how easy it is to understand text2 compared to text1. A high score means "
        "text2 retains essential points, is clear and concise, and does not lose important "
        "context, making it equally or more understandable than text1."
    ),
}


rubric = """Please score out of 20 points,
 using only zero or positive integers
 for each different aspect in the evaluation criteria"""
response_format = "output the scores in a JSON structured response"

# You are a helpful assistant and a judge to evaluate natural language texts.
# You will be given two text segments in the format: [text1][text2].
# For text2, your task is to provide integer scores (0-20) for each of the criteria below:
# {criteria}
system_prompt_builder_callback = lambda **system_kwargs: PromptTemplate(
    template="""You are a helpful assistant and a judge to evaluate natural language texts.
 You will be given two text segments in the following format: [text1][text2].
 For text2, your task is to provide a score integer scores (0-20) for each of the criteria below:
 {criteria}
 {rubric} and {response_format}.""",
    input_variables=["criteria", "rubric", "response_format"],
    partial_variables={},
)

user_prompt_builder_callback = lambda **user_kwargs: PromptTemplate(
    template="""
[text1: "{text_original}"] 
[text2: "{text_privatized}"]
""",
    input_variables=["text_original", "text_privatized"],
    partial_variables={
        "format_instructions": user_kwargs["output_parser"].get_format_instructions()
    },
)


class LLMSenseMetric:
    def __init__(self, llm_instance, *, criteria_path=None, retry_max_attempts=10):
        """Initialize the metric with the supported evaluation criteria.

        Args:
            criteria_path: path to a JSON document with the mapping between aspects and descriptions
        """
        self.llm_instance = llm_instance
        self.retry_max_attempts = retry_max_attempts

        if criteria_path and os.path.isfile(criteria_path):

            logger.info(f"Criteria file is found: {criteria_path}")
            with open(criteria_path) as criteria_file:
                self._CRITERIA = json.load(criteria_file)
        else:

            logger.warning(
                f"Criteria file is not found: {criteria_path}. Using default criteria."
            )
            self._CRITERIA = DEFAULT_CRITERIA

        logger.info(f"Defined aspects: {self._CRITERIA.keys()}")

    def get_prompter(
        self, system_prompt_builder_callback, user_prompt_builder_callback
    ) -> Tuple[OutputFixingParser, PromptTemplate, PromptTemplate]:
        """Construct and return an output parser and prompt templates."""
        
        output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=SenseScores),
            llm=self.llm_instance,
        )
        return (
            output_parser,
            system_prompt_builder_callback(**{}),
            user_prompt_builder_callback(**{"output_parser": output_parser}),
        )

    def _resolve_criteria(
        self, aspects: Optional[Union[Dict[str, str], List[str]]]
    ) -> Dict[str, str]:
        """Resolve the evaluation criteria to use in the prompt."""

        logger.debug(f"Input Aspects: {aspects}")

        criteria_: Dict[str, str] = dict()
        if aspects is None:
            criteria_ = self._CRITERIA
        elif isinstance(aspects, list):  # aspects = [x, y, z] as a list
            for aspect in aspects:
                if aspect in self._CRITERIA:  # presence in the predfined criteria
                    criteria_[aspect] = self._CRITERIA[aspect]
                else:
                    raise ValueError(
                        f"Aspect {aspect} is not defined. "
                        f"Please pass a path to JSON file or a dictionary with aspect to evaluation criteria."
                        f"E.g., {aspect}: description of the evaluation criteria"
                    )
        elif isinstance(aspects, dict):  # aspect to description mapping as a dict
            logger.debug(f"Using custom criteria passed at runtime. {aspects}")
            criteria_ = aspects
        else:
            raise ValueError(
                "Aspects cannot be empty. "
                "Please provide a aspect name or a mapping of the aspect name to its description."
            )

        return criteria_

    def compute_scores(
        self, texts_original, texts_privatized, aspects, batch_size=1
    ) -> List[Union[BaseModel, None]]:
        """Score a privatized text against the original using the LLM."""
        if isinstance(texts_original, str):
            texts_original = [texts_original]
        if isinstance(texts_privatized, str):
            texts_privatized = [texts_privatized]

        criteria = self._resolve_criteria(aspects)
        criteria_str = "\n".join(f"{k}: {v}" for k, v in criteria.items())

        output_parser, system_template, generation_template = self.get_prompter(
            system_prompt_builder_callback, user_prompt_builder_callback
        )
        
        messages = [
            [
                SystemMessage(
                    content=system_template.format(
                        criteria=criteria_str, rubric=rubric, response_format=response_format,
                    )
                ),
                HumanMessage(
                    content=generation_template.format(
                        text_original=orig, text_privatized=priv
                    )
                ),
            ] for orig, priv in zip(texts_original, texts_privatized) 
        ]
        
        indexed_messages = list(enumerate(messages))
        
        indexed_messages.sort(key=lambda x: len(x[1][1].content))
        
        final_scores = [None] * len(messages)
        
        i = 0
        pbar = tqdm(total=len(indexed_messages), desc="Scoring Batches")
        
        while i < len(indexed_messages):
            current_batch_size = min(batch_size, len(indexed_messages) - i)
            success = False
            
            while not success and current_batch_size > 0:
                batch_chunk = indexed_messages[i:i + current_batch_size]
                
                batch_orig_indices = [item[0] for item in batch_chunk]
                batch_messages = [item[1] for item in batch_chunk]
                
                batch_scores = [None] * len(batch_messages)
                pending_indices = list(range(len(batch_messages)))
                retry_cnt = 0
                
                try:
                    while pending_indices and retry_cnt < self.retry_max_attempts:
                        current_messages_to_process = [batch_messages[idx] for idx in pending_indices]
                        
                        output_texts = self.llm_instance.tensor_batch_generate(current_messages_to_process)
                        
                        new_pending_indices = []
                        for j, out_text in enumerate(output_texts):
                            local_idx = pending_indices[j]
                            try:
                                parsed_score = output_parser.parse(out_text)
                                batch_scores[local_idx] = parsed_score
                            except OutputParserException:
                                new_pending_indices.append(local_idx)
                                
                        pending_indices = new_pending_indices
                        if pending_indices:
                            retry_cnt += 1

                    if pending_indices:
                        logger.error(f"Failed to parse {len(pending_indices)} items after {self.retry_max_attempts} attempts.")
                    
                    for local_idx, score in enumerate(batch_scores):
                        orig_idx = batch_orig_indices[local_idx]
                        final_scores[orig_idx] = score
                        
                    i += current_batch_size
                    pbar.update(current_batch_size)
                    success = True

                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        f"CUDA OOM caught. Clearing cache and halving batch size "
                        f"(from {current_batch_size} to {current_batch_size // 2})."
                    )
                    
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    current_batch_size //= 2
                    
                    if current_batch_size == 0:
                        bad_orig_idx = batch_orig_indices[0]
                        logger.error(f"Sequence too long! Skipping document at original index {bad_orig_idx}.")
                        
                        i += 1 
                        pbar.update(1)
                        success = True 
                        
        pbar.close()
        return final_scores


def load_jsonl_to_dict(jsonl_path: str, sampled_doc_ids: Optional[list] = None) -> Dict[str, Dict[str, str]]:
    """
    Load JSONL into a dictionary with dynamic sentence extraction when needed.
    """
    df = pd.read_json(jsonl_path, lines=True)

    if sampled_doc_ids is not None:
        df = df[df["documentID"].isin(sampled_doc_ids)].reset_index(drop=True)

    docs = {}
    for _, row in df.iterrows():
        documentID = row["documentID"]
        full_text = row["fullText"]
        docs[documentID] = {
            "fullText": full_text,
        }

    return docs


def main(args: argparse.Namespace) -> None:
    """Command-line entrypoint: run sense scoring over jsonl files and save output."""

    sampled_doc_ids = []

    if getattr(args, "sampled_data_filepath", None):
        sample = pd.read_json(args.sampled_data_filepath)
        sampled_doc_ids = sample['documentID'].values.tolist()

    original_docs = load_jsonl_to_dict(args.text_original_jsonl, sampled_doc_ids=sampled_doc_ids)
    privatized_docs = load_jsonl_to_dict(args.text_privatized_jsonl, sampled_doc_ids=sampled_doc_ids)

    doc_similarity_metric = MultiLingualDocumentMeaningSimilarityMetric()
    doc_similarity_dict = doc_similarity_metric.infer_meaning_similarity(
        original_docs, privatized_docs
    )
    doc_similarity_metric.unload_model()

    criteria = None
    if args.criteria_file is not None:
        with fopen(args.criteria_file, "r") as f:
            criteria = f.read()

    huggingface_llm = HuggingFaceLLMAdapter(model_id=args.model_path)
    huggingface_llm.load_model()
    llm_sense_metric = LLMSenseMetric(
        llm_instance=huggingface_llm.get_llm_instance(),
        criteria_path=args.criteria_file,
    )

    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    score_sums = {}
    score_counts = {}

    for field_name in SenseScores.model_fields.keys():
        score_sums[field_name] = 0
        score_counts[field_name] = 0

    llm_score_dict = {}
    default_scores_dict = {
        field_name: None for field_name in SenseScores.model_fields.keys()
    }

    all_original_texts = []
    all_privatized_texts = []
    doc_ids = []
    for doc_id, doc in tqdm(original_docs.items()):
        text_original = doc["fullText"]
        text_privatized = privatized_docs[doc_id]["fullText"]
        all_original_texts.append(text_original)
        all_privatized_texts.append(text_privatized)
        doc_ids.append(doc_id)

        # llm_scores = llm_sense_metric.compute_scores(
        #     text_original, text_privatized, criteria
        # )

        # llm_score_dict[doc_id] = (
        #     llm_scores.model_dump() if llm_scores is not None else default_scores_dict
        # )
        # for field_name, val in scores_dict.items():
        #     if val is not None:
        #         score_sums[field_name] += val
        #         score_counts[field_name] += 1

        # logger.info(f"Document ID: {doc_id} - Scores: {scores_dict}")
        # print(doc_id, scores_dict)
        # f.write(
        #     json.dumps({
        #         "documentID": doc_id,
        #         "llm_scores": scores_dict
        #     }) + '\n'
        # )
        # break
    # with fopen(args.output_path.replace(".json", ".averages.json"), 'w') as f:
    #     averages = {}
    #     for field_name in score_sums:
    #         count = score_counts[field_name]
    #         if count > 0:
    #             # Compute average
    #             averages[field_name] = score_sums[field_name] / count
    #         else:
    #             # If no valid scores for that field, store 0 or None
    #             averages[field_name] = 0

    #     # -----------------------------
    #     # 4) Write the average scores as a final JSON line
    #     # -----------------------------
    #     f.write(json.dumps(averages) + "\n")

    # torch.cuda.empty_cache()  # Clear unused memory
    llm_scores = llm_sense_metric.compute_scores(
        all_original_texts, all_privatized_texts, criteria, batch_size=args.batch_size
    )

    for idx, doc_id in enumerate(doc_ids):
        llm_score_dict[doc_id] = (
            llm_scores[idx].model_dump() if llm_scores[idx] is not None else default_scores_dict
        )
    huggingface_llm.unload_model()

    llm_scores_df = pd.DataFrame.from_dict(llm_score_dict, orient="index")
    doc_similarity_df = pd.DataFrame.from_dict(doc_similarity_dict, orient="index")
    doc_similarity_df.rename(columns={0: "cosine_similarity"}, inplace=True)
    merged_df = pd.merge(
        llm_scores_df, doc_similarity_df, left_index=True, right_index=True, how="outer"
    )
    merged_df = merged_df.reset_index().rename(columns={"index": "documentID"})
    merged_df.to_json(args.output_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-original-jsonl", type=str)
    parser.add_argument("--text-privatized-jsonl", type=str)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for LLM inference")
    parser.add_argument(
        "--output-path", type=str, help="Path to save the generated JSONL file"
    )
    parser.add_argument("--model-path", type=str, help="Path to model", default="e")
    parser.add_argument(
        "--criteria-file", type=str, help="Path to criteria file", default=None
    )
    parser.add_argument(
        "--do-sample", help="Use sampling for generation", action="store_true"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampled-data-filepath", type=str, default=None)
    args = parser.parse_args()
    pprint.pprint(vars(args))
    # logger.info("Arguments:")
    # logger.info(args)

    # logger.info("do_sample: ", args.do_sample, "temperature: ", args.temperature)

    generation_config = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
    }
    # if os.path.exists(args.output_path) or os.path.exists(args.output_path.replace(".json", ".averages.json")):
    #     exit(f"Output file already exists: {args.output_path}")
    main(args)