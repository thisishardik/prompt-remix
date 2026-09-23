from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from hiatus.evaluation.model_registry import ModelConfig, get_model_config


class EmbeddingExtractor:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_memory_mb: int = 40000,
        use_flash_attention: bool = False,
        quantize_4bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_memory_mb = max_memory_mb
        self.use_flash_attention = use_flash_attention
        self.quantize_4bit = quantize_4bit

        try:
            self.config = get_model_config(model_name)
        except KeyError:
            print(f"Model '{model_name}' not in AuthBench registry.")
            self.config = ModelConfig(
                hf_path=model_name,
                model_type="encoder",
                pooling="mean",
            )

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        cfg = self.config

        if cfg.model_type == "sentence_transformer":
            self._load_sentence_transformer()
        elif cfg.model_type == "encoder":
            self._load_encoder()
        elif cfg.model_type == "causal_lm":
            self._load_causal_lm()
        else:
            raise ValueError(f"Unknown model type: {cfg.model_type}")

    def _load_sentence_transformer(self):
        self.model = SentenceTransformer(
            self.config.hf_path,
            device=self.device,
            trust_remote_code=self.config.trust_remote_code,
        )
        print(f"Loaded sentence-transformer: {self.config.hf_path} on {self.device}")

    def _load_encoder(self):
        self.model = AutoModel.from_pretrained(
            self.config.hf_path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loaded encoder: {self.config.hf_path} on {self.device}")

    def _load_causal_lm(self):
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": self.config.trust_remote_code,
        }

        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        if self.quantize_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_path, **model_kwargs
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_path,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loaded causal LM: {self.config.hf_path}")

    def _mean_pool(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(
            min=1e-9
        )

    def _cls_pool(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        return last_hidden_state[:, 0]

    def _last_token_pool(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.pooling == "mean":
            return self._mean_pool(last_hidden_state, attention_mask)
        elif self.config.pooling == "cls":
            return self._cls_pool(last_hidden_state)
        elif self.config.pooling == "last_token":
            return self._last_token_pool(last_hidden_state, attention_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling}")

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> NDArray[np.float64]:
        if self.config.model_type == "sentence_transformer":
            return self._encode_sentence_transformer(texts, batch_size, show_progress)
        else:
            return self._encode_hf(texts, batch_size, show_progress)

    def _encode_sentence_transformer(
        self, texts: List[str], batch_size: int, show_progress: bool
    ) -> NDArray[np.float64]:
        if not self.config.chunking:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float64)
        else:
            all_embeddings = []
            max_len = getattr(self.model, "max_seq_length", self.config.max_length)
            if max_len is None or max_len <= 0:
                max_len = 512
                
            tokenizer = self.model.tokenizer
            chunk_size = max_len - 2  # reserve space for special tokens
            
            iterator = tqdm(texts, desc=f"Encoding ({self.model_name})") if show_progress else texts
            
            for text in iterator:
                if not text.strip():
                    text = " "
                tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
                if len(tokens) == 0:
                    chunks = [text]
                else:
                    chunk_tokens = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
                    chunks = [tokenizer.decode(c, skip_special_tokens=True) for c in chunk_tokens]
                
                chunk_embs = self.model.encode(
                    chunks, 
                    batch_size=batch_size, 
                    show_progress_bar=False, 
                    normalize_embeddings=True, 
                    convert_to_numpy=True
                )
                
                doc_emb = chunk_embs.mean(axis=0)
                doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-12)
                all_embeddings.append(doc_emb)
                
            return np.array(all_embeddings, dtype=np.float64)

    def _encode_hf(
        self, texts: List[str], batch_size: int, show_progress: bool
    ) -> NDArray[np.float64]:
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Encoding ({self.model_name})")

        for i in iterator:
            # Replace empty strings with a space to avoid 0-length token sequences which crash Qwen2
            batch_texts = [text if text.strip() else " " for text in texts[i : i + batch_size]]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )

            if self.config.model_type == "causal_lm":
                target_device = next(self.model.parameters()).device
            else:
                target_device = self.device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            outputs = self.model(**inputs, output_hidden_states=True)

            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states[-1]

            pooled = self._pool(hidden_states, inputs["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu().float().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings.astype(np.float64)

    def unload(self):
        del self.model
        self.model = None
        del self.tokenizer
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Unloaded model: {self.model_name}")
