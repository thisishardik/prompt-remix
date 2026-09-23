import json
import os
import pprint
import time
import warnings
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Optional, Tuple

import openai
import pandas as pd
import torch
import tqdm
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


@dataclass
class ParaphrasePrompt:
    """Container for 'less' and 'more' paraphrase prompt templates."""

    less: str
    more: str


class Paraphraser(ABC):
    """Abstract base class for paraphrasers.
    Subclasses implement `make_completions` to generate paraphrases.
    """

    def __init__(self, style: str, model_name: str, prompts: ParaphrasePrompt):
        """Save initialization parameters."""
        self.prompts = prompts
        self.model_name = model_name
        self.style = style

    def paraphrase(self, df: pd.DataFrame) -> pd.DataFrame:
        """Paraphrase texts in DataFrame and add new columns."""
        df = df.copy()
        less, more = self.paraphrase_prompts(df["original"])

        if less is not None:
            df[f"{self.style}_less"] = less

        if more is not None:
            df[f"{self.style}_more"] = more

        return df

    def paraphrase_prompts(
        self, texts: List[str]
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """Build user prompts and return lists of paraphrases for 'less' and 'more'."""
        user_prompts = []
        less, more = None, None

        if "jais-adapted" in self.model_name.lower():
            user_prompts = [
                f"النص: {text}\nأعد صياغته بأسلوب مختلف مع الحفاظ على المعنى. "
                f"احرص على أن تكون الإجابة بالعربية فقط دون ترجمة إلى الإنجليزية."
                for text in texts
            ]
        else:
            user_prompts = [f"Paragraph: {text} \n Rewrite:" for text in texts]

        if self.prompts.less != "":
            less = self.make_completions(self.prompts.less, user_prompts)

        if self.prompts.more != "":
            more = self.make_completions(self.prompts.more, user_prompts)

        return less, more

    @abstractmethod
    def make_completions(self, prompt: str, texts: List[str]) -> List[str]:
        """Generate paraphrase completions for each text using `prompt`."""
        pass


class OpenAIParaphraser(Paraphraser):
    """Paraphraser that uses the OpenAI client to create chat completions."""

    def __init__(self, style: str, model_name: str, prompts: ParaphrasePrompt):
        super().__init__(style, model_name, prompts)
        self.client = openai.Client()

    def make_completions(self, prompt: str, texts: List[str]) -> List[str]:
        """Call the OpenAI client to paraphrase each text."""
        completions = []
        for text in tqdm.tqdm(texts, desc=f"Paraphrasing"):
            completions.append(self.make_completion(prompt, text))
        return completions

    def make_completion(self, prompt: str, text: str) -> str:
        """Create a single completion for `text` using the OpenAI chat API."""
        messages = [
            {"role": "developer", "content": prompt},
            {"role": "user", "content": text},
        ]
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=messages
        )
        return completion.choices[0].message["content"]


class LocalParaphraser(Paraphraser):
    """Paraphraser that runs a local LLM via vLLM for generation."""

    def __init__(self, style: str, model_name: str, prompts: ParaphrasePrompt):
        """Initialize vLLM model, tokenizer, and sampling parameters."""
        super().__init__(style, model_name, prompts)
        num_gpus = torch.cuda.device_count()
        max_model_len = (
            4096 if "jais-adapted" in self.model_name.lower() else 10000
        )

        print(f"Using {num_gpus} GPUs")

        # Make max_model_len configurable
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=num_gpus,
            max_model_len=max_model_len,
        )
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            top_k=40,
            max_tokens=8000,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        if "jais-adapted" in self.model_name.lower():
            self.sampling_params = SamplingParams(
                n=1,
                temperature=0.55,
                top_p=0.9,
                repetition_penalty=1.1,
                top_k=50,
                max_tokens=800,
            )
            self.tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}
                                    ### Instruction: {{ messages[0]['content'] }}
                                    Complete the conversation below between [|Human|] and [|AI|]:
                                    ### Input:
                                    {% set loop_messages = messages[1:] %}
                                    {% else %}
                                    ### Instruction: Your name is 'Jais', and you are a helpful assistant.
                                    Complete the conversation below between [|Human|] and [|AI|]:
                                    ### Input:
                                    {% set loop_messages = messages %}
                                    {% endif %}
                                    {% for message in loop_messages %}
                                    [|{{ message['role']|capitalize }}|]: {{ message['content'] }}
                                    {% endfor %}
                                    [|AI|]:
                                    """
        else:
            self.tokenizer.chat_template = """{% for message in messages %}
                                    {% if message['role'] == 'system' %}
                                    {{ message['content'] }}

                                    {% elif message['role'] == 'user' %}
                                    {{ message['content'] }}

                                    {% endif %}
                                    {% endfor %}"""

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def make_completions(self, prompt: str, texts: List[str]) -> List[str]:
        """Generate paraphrases locally using the vLLM engine."""

        if "jais-adapted" in self.model_name.lower():
            prompt = prompt.strip() + "\n\n يرجى أن تكون الإجابة باللغة العربية فقط"

        chats = [
            [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
            for text in texts
        ]

        add_generation_prompt = not (
            "jais-adapted" in self.model_name.lower()
        )

        prompts = [
            self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            for chat in chats
        ]

        outputs = self.llm.generate(
            prompts=prompts, sampling_params=self.sampling_params, use_tqdm=True
        )

        completions = [output.outputs[0].text.strip() for output in outputs]
        return completions

    def postprocess_completion(self, text: str) -> str:
        """Strip assistant header if present and return cleaned text."""
        text = text.strip()
        if text.startswith("assistant\n\n"):
            text = text[len("assistant\n\n") :]
        if text.startswith("[|AI|]:"):
            text = text.replace("[|AI|]:", "").strip()
        return text


class HuggingFaceParaphraser(Paraphraser):

    # google/mt-small -> specify a pad token & remove all special tokens during generation

    def __init__(self, style: str, model_name: str, prompts: ParaphrasePrompt):
        super().__init__(style, model_name, prompts)
        num_gpus = torch.cuda.device_count()
        self.device = "cuda:0" if num_gpus > 0 else "cpu"
        print(f"Using {num_gpus} GPUs")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", trust_remote_code=True
        )
        # self.model.to(self.device)
        self.model.eval()
        print(f"Running model on {self.model.device}")
        self.max_length = 512
        self.max_new_tokens = 128
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        if self.model_name.lower().__contains__("jais"):
            self.tokenizer.chat_template = """{% for message in messages %}
                                    {% if message['role'] == 'system' %}
                                    <|system|>
                                    {{ message['content'] }}
                                    <|end|>
                                    {% elif message['role'] == 'user' %}
                                    <|user|>
                                    {{ message['content'] }}
                                    <|end|>
                                    {% elif message['role'] == 'assistant' %}
                                    <|assistant|>
                                    {{ message['content'] }}
                                    <|end|>
                                    {% endif %}
                                    {% endfor %}
                                    <|assistant|>"""
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.chat_template = """{% for message in messages %}
                                            {% if message['role'] == 'system' %} ### Instruction: {{ message['content'] }}
                                            {% elif message['role'] == 'user' %} {{ message['content'] }}
                                            {% endif %}
                                            {% endfor %}"""

    def make_completions(self, prompt: str, texts: List[str]) -> List[str]:
        chats = [
            [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
            for text in texts
        ]

        tokenized_prompts_batch = self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=False,
            padding=True,
            truncation=True,
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.device)

        cleaned_responses = []

        t0 = time.time()

        for input_ids, attention_mask in tqdm.tqdm(
            zip(
                tokenized_prompts_batch.input_ids,
                tokenized_prompts_batch.attention_mask,
            ),
            total=len(tokenized_prompts_batch),
        ):
            input_ids = input_ids.unsqueeze(0)  # 1 x 256
            attention_mask = attention_mask.unsqueeze(0)
            input_ids = input_ids.to(self.device)
            generated_ids = None

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    repetition_penalty=1.05,
                    max_new_tokens=self.max_new_tokens,
                    max_length=self.max_length,
                )

            cleaned_ids = None

            if torch.equal(generated_ids[0][: input_ids[0].size(0)], input_ids[0]):
                cleaned_ids = generated_ids[0][input_ids[0].size(0) :]
            else:
                cleaned_ids = generated_ids[0]

            response = self.tokenizer.decode(
                cleaned_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            cleaned_responses.append(response.strip())

        print(
            f"Paraphrasing inference completed: [STYLE: {self.style}] - [DEVICE: {self.device}] - [TOTAL_TIME: {(time.time() - t0):.3f} secs] - [AVG TIME/RECORD: {(time.time()-t0) / len(tokenized_prompts_batch):.3f} secs]"
        )
        return cleaned_responses

    def postprocess_completion(self, text: str) -> str:
        if text.startswith("assistant\n\n"):
            text = text[len("assistant\n\n") :]
        return text


def parse_prompts(prompts: str) -> Tuple[str, ParaphrasePrompt]:
    """Load prompts from a .j2 or .json file and return (style, ParaphrasePrompt)."""
    style, prompt_ext = os.path.splitext(os.path.basename(prompts))
    if prompt_ext == ".j2":
        with open(prompts, "r") as f:
            template = Template(f.read())
        prompts = json.loads(template.render())
    elif prompt_ext == ".json":
        with open(prompts, "r") as f:
            prompts = json.load(f)
    else:
        raise ValueError(
            f"Unsupported prompt format: {prompt_ext}. Supported formats are .j2, .json"
        )
    return style, ParaphrasePrompt(**prompts)


def main():
    """Command-line entrypoint: parse args, load prompts, and run paraphrasing."""
    parser = ArgumentParser()
    parser.add_argument("--use_openai", action="store_true")
    parser.add_argument(
        "--model-name",
        type=str,
        default="neuralmagic-ent/Llama-3.3-70B-Instruct-quantized.w8a8",
    )  # RU: msu-rcc-lair/RuadaptQwen2.5-32B-Instruct
    parser.add_argument(
        "--prompts", type=str, default="pre_obfuscation/prompts_zh/sarcasm.j2"
    )
    parser.add_argument(
        "--adapter-base-texts",
        type=str,
        default="/gscratch/amath/kogolobo/StyleRemix/data/en_all_samples_adapter.jsonl",
    )
    parser.add_argument(
        "--classifier-base-texts",
        type=str,
        default="/gscratch/amath/kogolobo/StyleRemix/data/en_all_samples_non_adapter.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/gscratch/amath/kogolobo/StyleRemix/data/"
    )
    args = parser.parse_args()
    pprint.pprint(vars(args))

    style, prompts = parse_prompts(args.prompts)

    os.makedirs(args.output_dir, exist_ok=True)

    output_path_adapter = os.path.join(
        args.output_dir, f"{style}_adapter_examples.jsonl"
    )
    output_path_classifier = os.path.join(
        args.output_dir, f"{style}_classifier_examples.jsonl"
    )
    need_adapter = os.path.exists(output_path_adapter) and args.adapter_base_texts != ""
    need_classifier = (
        not os.path.exists(output_path_classifier) and args.classifier_base_texts != ""
    )
    if not need_adapter and not need_classifier:
        print("All files already exist, nothing to do")
        return

    paraphraser_cls = OpenAIParaphraser if args.use_openai else LocalParaphraser

    paraphraser = paraphraser_cls(style, args.model_name, prompts)

    if need_adapter:
        adapter_base_texts = pd.read_json(args.adapter_base_texts, lines=True)
        adapter_paraphrased = paraphraser.paraphrase(adapter_base_texts)
        adapter_paraphrased.to_json(
            output_path_adapter, lines=True, orient="records", force_ascii=False
        )

    if need_classifier:
        classifier_base_texts_df = None
        try:
            classifier_base_texts_df = pd.read_json(
                args.classifier_base_texts, lines=True
            )
        except Exception as e:
            with open(args.classifier_base_texts, "r") as handle:
                classifier_base_texts = handle.read()
            if isinstance(classifier_base_texts, str):
                classifier_base_texts_df = pd.DataFrame(
                    json.loads(classifier_base_texts)
                )

        classifier_paraphrased = paraphraser.paraphrase(classifier_base_texts_df)
        classifier_paraphrased.to_json(
            output_path_classifier, lines=True, orient="records", force_ascii=False
        )


if __name__ == "__main__":
    main()
