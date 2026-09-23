import random
from typing import Dict, List, Optional, Tuple
import torch
from vllm import LLM, SamplingParams
from vllm.config import StructuredOutputsConfig
from transformers import AutoTokenizer

style_strengths_en = {0.5: "weak", 0.7: "medium", 0.9: "strong", 1.0: "very strong"}
style_strengths_ru = {
    0.5: "слабый",
    0.7: "умеренный",
    0.9: "сильный",
    1.0: "очень сильный",
}
style_strengths_zh = {0.5: "弱", 0.7: "中等", 0.9: "强", 1.0: "非常强"}
style_strengths_ar = {0.5: "ضعيف", 0.7: "متوسط", 0.9: "قوي", 1.0: "قوي جدًا"}

class PromptRemixRunner:
    def __init__(
        self,
        model_id: str,
        prompt_dict: Dict[str, str],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        lang: str = "en",
        decoding_strategy: str = "default",
        max_gpu_memory_utilization: float = 0.95,
        seed: int = 42,
    ):
        self.lang = lang
        self.style_strength_dict = self._get_style_strength(lang)
        self.prompt_dict = prompt_dict
        self.model_name = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.decoding_strategy = decoding_strategy
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.max_gpu_memory_utilization = max_gpu_memory_utilization

        # 1. Fetch unified token budgets
        self.input_tokens_per_doc, self.reasoning_tokens, self.output_tokens = self._get_token_budgets()
        
        # 2. Calculate absolute Max Model Length (capable of holding up to 10 docs + reasoning + output)
        self.max_model_len = (10 * self.input_tokens_per_doc) + self.reasoning_tokens + self.output_tokens
        
        num_gpus = torch.cuda.device_count()
        self.device = "cuda" if num_gpus > 0 else "cpu"
        print(f"Using {num_gpus} GPUs. Max model len set to: {self.max_model_len}")

        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=num_gpus,
            max_model_len=self.max_model_len,
            language_model_only=True,
            trust_remote_code=True,
            gpu_memory_utilization=self.max_gpu_memory_utilization,
            structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        )

        self._setup_chat_template()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._init_sampling_params()

    def _get_token_budgets(self) -> Tuple[int, int, int]:
        """
        Returns (input_tokens_per_doc, reasoning_tokens, output_tokens).
        Total max_model_len is calculated as: (10 * input_tokens_per_doc) + reasoning + output.
        """
        model_lower = self.model_name.lower()
        
        if "qwen3.5-35b" in model_lower or "qwen3.5-27b" in model_lower:
            return 2000, 20000, 2000  # Total: 42,000
        elif "qwen3.5" in model_lower:
            return 2000, 20000, 2000  # Total: 42,000
        elif "qwen" in model_lower:
            return 400, 2000, 2000    # Total: 8,000
        elif "jais-adapted-70b-chat" in model_lower:
            return 100, 1096, 2000    # Total: 4,096
        elif "acegpt" in model_lower or "jais-2" in model_lower:
            return 419, 2000, 2000    # Total: 8,190 (~8192)
        elif "llama" in model_lower:
            return 400, 400, 2000     # Total: 6,400
        else:
            return 600, 2000, 2000    # Default: 10,000

    def _get_style_strength(self, lang):
        if lang == "en": return style_strengths_en
        if lang == "ru": return style_strengths_ru
        if lang == "zh": return style_strengths_zh
        if lang == "ar": return style_strengths_ar
        raise ValueError(f"Unsupported language: {lang}")

    def _setup_chat_template(self):
        if "jais-adapted" in self.model_name.lower():
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

    def _init_sampling_params(self):
        base_kwargs = self._build_base_sp_kwargs()

        # 1. Unconstrained decoding runs on the full budget (reasoning + output)
        unconstrained_kwargs = base_kwargs.copy()
        unconstrained_kwargs["max_tokens"] = self.reasoning_tokens + self.output_tokens
        self.base_sampling_params = SamplingParams(**unconstrained_kwargs)
        
        # 2. Reasoning logic runs purely on reasoning tokens
        reasoning_kwargs = base_kwargs.copy()
        reasoning_kwargs["max_tokens"] = self.reasoning_tokens
        reasoning_kwargs["stop"] = ["</think>"]
        self.reasoning_sampling_params = SamplingParams(**reasoning_kwargs)

        # 3. Constrained phase runs purely on output tokens
        constrained_kwargs = base_kwargs.copy()
        constrained_kwargs["max_tokens"] = self.output_tokens
        
        try:
            from vllm.sampling_params import StructuredOutputsParams
            regex = self._get_regex(self.lang)
            structured_outputs = StructuredOutputsParams(regex=regex)
            constrained_kwargs["structured_outputs"] = structured_outputs
            print(f"Constrained decoding configured for {self.lang} using llguidance API")
        except ImportError as e:
            print(f"Failed to import StructuredOutputsParams: {e}")
            
        self.constrained_sampling_params = SamplingParams(**constrained_kwargs)
        
    def _build_base_sp_kwargs(self) -> Dict:
        """Returns clean sampling defaults. max_tokens is now handled in _init_sampling_params."""
        strategy = self.decoding_strategy.lower()
        sp = dict(
            n=1,
            seed=self.seed,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )

        model_lower = self.model_name.lower()
        if "llama" in model_lower:
            sp.update(skip_special_tokens=True, spaces_between_special_tokens=False)
        elif "qwen3.5" in model_lower:
            sp.update(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0)
        elif "qwen" in model_lower:
            sp.update(temperature=0.6, top_p=0.95, top_k=40, repetition_penalty=1.05)

        if strategy == "temperature":
            sp.update({"top_p": 1.0, "top_k": -1, "temperature": self.temperature})
        elif strategy == "greedy":
            sp.update({"temperature": 0.0, "top_k": -1, "top_p": 1.0})
        elif strategy == "top_k":
            sp.update({"top_p": 1.0, "top_k": self.top_k, "temperature": 1.0})
        elif strategy in ["top_p", "nucleus"]:
            sp.update({"top_p": self.top_p, "top_k": -1, "temperature": 1.0})
            
        return sp

    def _get_regex(self, lang: str) -> str:
        if lang == "ar":
            return r'''[\x{0600}-\x{06FF}\x{0750}-\x{077F}\x{08A0}-\x{08FF}\x{FB50}-\x{FDFF}\x{FE70}-\x{FEFF}0-9\x{0660}-\x{0669}\s،؛؟.,:;!%\-\+()\[\]{}/"'«»]+'''
        elif lang == "en":
            return r'''[A-Za-z0-9\s.,:;!?%\-\+()\[\]{}/"'’]+'''
        elif lang == "ru":
            return r'''[\x{0400}-\x{04FF}0-9\s.,:;!?%\-\+()\[\]{}/"'«»]+'''
        elif lang == "zh":
            return r'''[\x{4E00}-\x{9FFF}\x{3400}-\x{4DBF}0-9\s。，、！？：；,.!?]+'''
        return r".+"

    def _get_transition_phrase(self, lang: str) -> str:
        phrases = {
            "en": "I will produce the final response in English.",
            "ru": "Я предоставлю итоговый ответ на русском.",
            "zh": "我将用中文提供最终回复。",
            "ar": "سأقوم بإنشاء الرد النهائي باللغة العربية."
        }
        return phrases.get(lang, "I will produce the final response in the specified language.")

    def normalize_directions(self, directions: Dict[str, float]):
        norm_directions = {}
        for direction, weight in directions.items():
            if weight is not None and weight != 0:
                if "type" in direction:
                    norm_directions[direction] = abs(weight)
                elif weight < 0:
                    direction = direction.replace("more", "less")
                    norm_directions[direction] = abs(weight)
                else:
                    norm_directions[direction] = weight
        return norm_directions

    def _extract_genre_summary(self, genre_examples: List[str], use_constrained_decoding: bool) -> str:
        """Derives a single-sentence genre summary from up to 10 randomly sampled examples."""
        # Randomly sample up to 10 examples
        if len(genre_examples) > 10:
            sampled_examples = random.sample(genre_examples, 10)
        else:
            sampled_examples = genre_examples

        extraction_template = self.prompt_dict.get(
            "genre_extraction", 
            "Analyze the following text examples and write a 1-sentence description of their shared genre:\n{examples}"
        )
        
        valid_examples = []
        separator = "\n\n---\n\n"
        sep_tokens = len(self.tokenizer.encode(separator))
        
        for ex in sampled_examples:
            ex_tokens = self.tokenizer.encode(ex)
            # Truncate strictly to the per-document budget to guarantee OOM safety
            if len(ex_tokens) + sep_tokens > self.input_tokens_per_doc:
                encoded_ex = ex_tokens[:self.input_tokens_per_doc - sep_tokens]
                truncated_ex = self.tokenizer.decode(encoded_ex, skip_special_tokens=True)
                valid_examples.append(truncated_ex)
            else:
                valid_examples.append(ex)

        examples_text = separator.join(valid_examples)
        extraction_prompt = extraction_template.replace("{examples}", examples_text)
        
        messages = [[{"role": "user", "content": extraction_prompt}]]
        
        summary = self._generate_responses(messages, use_constrained_decoding)[0]
        return summary.strip()

    def craft_instruction(self, directions: Dict[str, float], genre_summary: Optional[str] = None) -> str:
        prompt = self.prompt_dict["preamble"] + "\n\n"
        
        if genre_summary:
            genre_consistency_template = self.prompt_dict.get("genre_consistency", "Maintain the genre: {genre_summary}")
            prompt += genre_consistency_template.replace("{genre_summary}", genre_summary) + "\n\n"

        for idx, (direction, weight) in enumerate(directions.items()):
            if weight is not None and weight > 0:
                style_strength = self.style_strength_dict.get(weight, "medium")
                direction_prompt = self.prompt_dict.get(direction).replace(
                    "##style_strength##", style_strength
                )
                prompt += f"{idx+1}. {direction_prompt}\n\n"
        prompt += self.prompt_dict["postamble"]

        if "jais-adapted" in self.model_name.lower():
            prompt = prompt.strip() + "\n\n يرجى أن تكون الإجابة باللغة العربية فقط"

        return prompt

    def _prepare_prompts(self, messages: List[List[Dict[str, str]]]) -> List[str]:
        prompts = []
        for message in messages:
            add_generation_prompt = True
            if any(m in self.model_name.lower() for m in ["llama", "jais-adapted", "acegpt"]):
                add_generation_prompt = False
            
            prompt = self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            prompts.append(prompt)
        return prompts

    def _postprocess_completion(self, text: str) -> str:
        text = text.strip()
        if text.startswith("assistant\n\n"):
            text = text[len("assistant\n\n"):]
        if "/think>" in text:
            text = text.split("/think>", maxsplit=1)[-1].strip()
        return text

    def _generate_responses(self, messages: List[List[Dict[str, str]]], use_constrained_decoding: bool) -> List[str]:
        """Unified abstraction for executing the vLLM inference generation."""
        prompts = self._prepare_prompts(messages)
        
        if not use_constrained_decoding:
            outputs = self.llm.generate(
                prompts=prompts, sampling_params=self.base_sampling_params, use_tqdm=True
            )
            return [self._postprocess_completion(out.outputs[0].text) for out in outputs]
            
        reasoning_outputs = self.llm.generate(
            prompts=prompts, sampling_params=self.reasoning_sampling_params, use_tqdm=True
        )
        
        transition_phrase = self._get_transition_phrase(self.lang)
        constrained_prompts = []
        for prompt, out in zip(prompts, reasoning_outputs):
            trace = out.outputs[0].text
            new_prompt = f"{prompt}{trace}\n{transition_phrase}\n</think>\n"
            constrained_prompts.append(new_prompt)

        final_outputs = self.llm.generate(
            prompts=constrained_prompts, sampling_params=self.constrained_sampling_params, use_tqdm=True
        )
        return [out.outputs[0].text.strip() for out in final_outputs]

    def remix(
        self, 
        texts: List[str], 
        directions: List[Dict[str, float]], 
        use_constrained_decoding: bool = False,
        genre_examples: Optional[List[str]] = None
    ) -> List[str]:
        genre_summary = None
        if genre_examples and len(genre_examples) > 0:
            print(f"Extracting genre summary from examples...")
            genre_summary = self._extract_genre_summary(genre_examples, use_constrained_decoding)
            print(f"Extracted genre summary: {genre_summary}")

        norm_directions = [self.normalize_directions(dirs) for dirs in directions]
        instructions = [self.craft_instruction(dirs, genre_summary) for dirs in norm_directions]
        messages = [
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ]
            for text, instruction in zip(texts, instructions)
        ]

        return self._generate_responses(messages, use_constrained_decoding)