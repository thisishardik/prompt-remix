import asyncio
from functools import partial
from typing import List, Optional, Any, Dict, cast
from typing_extensions import override

import torch
from langchain_core.language_models.chat_models import BaseChatModel, BaseMessage, CallbackManagerForLLMRun, \
    AsyncCallbackManagerForLLMRun, ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel

from hiatus.evaluation import logger

class MyHuggingFaceModel(BaseChatModel):
    """
    https://github.com/langchain-ai/langchain/discussions/9596
    https://qwen.readthedocs.io/en/latest/framework/Langchain.html
    """
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.8
    history_len: int = 0
    huggingface_tokenizer: PreTrainedTokenizerBase
    huggingface_model: PreTrainedModel

    @property
    def param_inject_huggingface_generation(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            # "history_len": self.history_len
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return self.param_inject_huggingface_generation
    
    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        output_str = self.tensor_batch_generate([messages])[0]
        message = AIMessage(content=output_str)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def tensor_batch_generate(self, batched_messages: List[List[BaseMessage]]) -> List[str]:
        texts = []
        for messages in batched_messages:
            resolved_messages = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    resolved_messages.append({"role": "system", "content": message.content})
                elif isinstance(message, HumanMessage):
                    resolved_messages.append({"role": "user", "content": message.content})
                else:
                    raise NotImplementedError("Unknown type {}".format(type(message)))

            text = self.huggingface_tokenizer.apply_chat_template(
                resolved_messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        model_inputs = self.huggingface_tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(self.huggingface_model.device)

        self.huggingface_model.generation_config.eos_token_id = self.huggingface_tokenizer.eos_token_id
        self.huggingface_model.generation_config.pad_token_id = self.huggingface_tokenizer.pad_token_id

        generated_ids = self.huggingface_model.generate(
            **model_inputs,
            **self.param_inject_huggingface_generation
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = self.huggingface_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        cleaned = []
        for text in responses:
            text = text.replace("\u010a", "\n")
            text = text.replace("\u0120", " ")
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            cleaned.append(text)
        return cleaned

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        func = partial(self._generate, messages, stop=stop, run_manager=run_manager, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)


class HuggingFaceLLMAdapter:
    def __init__(self, model_id, *, cache_dir=None, local_files_only=False):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.is_load = False
        self.huggingface_model = None
        self.huggingface_tokenizer = None

    def load_model(self):
        self.huggingface_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            local_files_only=self.local_files_only,
            trust_remote_code=True, 
            torch_dtype="auto",
            device_map="auto"
        )
        self.huggingface_tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
            fix_mistral_regex=True
        )
        self.huggingface_tokenizer.padding_side = "left"
        if self.huggingface_tokenizer.pad_token is None:
            self.huggingface_tokenizer.pad_token = self.huggingface_tokenizer.eos_token

        self.huggingface_model.generation_config.pad_token_id = self.huggingface_tokenizer.pad_token_id
        self.is_load = True

    def unload_model(self):
        del self.huggingface_model
        self.huggingface_model = None
        del self.huggingface_tokenizer
        self.huggingface_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_load = False

    def get_llm_instance(self, **additional_kwargs):
        assert self.is_load is True
        return MyHuggingFaceModel(huggingface_tokenizer=self.huggingface_tokenizer,
                                  huggingface_model=self.huggingface_model, **additional_kwargs)
