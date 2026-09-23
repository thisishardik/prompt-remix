from dataclasses import dataclass
from typing import Dict, Literal


@dataclass
class ModelConfig:
    hf_path: str
    model_type: Literal["sentence_transformer", "encoder", "causal_lm"]
    pooling: Literal["mean", "cls", "last_token"]
    max_length: int = 512
    trust_remote_code: bool = False
    chunking: bool = False


AUTHBENCH_MODELS: Dict[str, ModelConfig] = {
    # Instruction-tuned LLMs
    "deepseek-coder-6.7b-instruct": ModelConfig(
        hf_path="deepseek-ai/deepseek-coder-6.7b-instruct",
        model_type="causal_lm",
        pooling="mean",
    ),
    "deepseek-llm-7b-chat": ModelConfig(
        hf_path="deepseek-ai/deepseek-llm-7b-chat",
        model_type="causal_lm",
        pooling="mean",
    ),
    "llama3-8b-instruct": ModelConfig(
        hf_path="meta-llama/Meta-Llama-3-8B-Instruct",
        model_type="causal_lm",
        pooling="mean",
    ),
    "llama3.1-8b-instruct": ModelConfig(
        hf_path="meta-llama/Llama-3.1-8B-Instruct",
        model_type="causal_lm",
        pooling="mean",
    ),
    "qwen2.5-3b-instruct": ModelConfig(
        hf_path="Qwen/Qwen2.5-3B-Instruct",
        model_type="causal_lm",
        pooling="mean",
        trust_remote_code=True,
    ),
    "qwen2.5-7b-instruct": ModelConfig(
        hf_path="Qwen/Qwen2.5-7B-Instruct",
        model_type="causal_lm",
        pooling="mean",
        trust_remote_code=True,
    ),
    "qwen3-4b-instruct": ModelConfig(
        hf_path="Qwen/Qwen3-4B",
        model_type="causal_lm",
        pooling="mean",
        trust_remote_code=True,
    ),
    # Base LLMs
    "deepseek-llm-7b-base": ModelConfig(
        hf_path="deepseek-ai/deepseek-llm-7b-base",
        model_type="causal_lm",
        pooling="mean",
    ),
    "llama3-8b": ModelConfig(
        hf_path="meta-llama/Meta-Llama-3-8B",
        model_type="causal_lm",
        pooling="mean",
    ),
    "llama3.1-8b": ModelConfig(
        hf_path="meta-llama/Llama-3.1-8B",
        model_type="causal_lm",
        pooling="mean",
    ),
    "qwen2.5-3b": ModelConfig(
        hf_path="Qwen/Qwen2.5-3B",
        model_type="causal_lm",
        pooling="mean",
        trust_remote_code=True,
    ),
    "qwen3-4b": ModelConfig(
        hf_path="Qwen/Qwen3-4B",
        model_type="causal_lm",
        pooling="mean",
        trust_remote_code=True,
    ),
    "llama2-7b": ModelConfig(
        hf_path="meta-llama/Llama-2-7b-hf",
        model_type="causal_lm",
        pooling="mean",
    ),
    "llama2-7b-chat": ModelConfig(
        hf_path="meta-llama/Llama-2-7b-chat-hf",
        model_type="causal_lm",
        pooling="mean",
    ),
    # Instruction-tuned embedding models
    "e5-mistral-7b-instruct": ModelConfig(
        hf_path="intfloat/e5-mistral-7b-instruct",
        model_type="causal_lm",
        pooling="last_token",
        max_length=4096,
    ),
    "gte-qwen2-7b-instruct": ModelConfig(
        hf_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        model_type="causal_lm",
        pooling="last_token",
        max_length=4096,
        trust_remote_code=True,
    ),
    "sfr-embedding-mistral": ModelConfig(
        hf_path="Salesforce/SFR-Embedding-Mistral",
        model_type="causal_lm",
        pooling="last_token",
        max_length=4096,
    ),
    # Base embedding models (sentence-transformer style)
    "multilingual-style-representation": ModelConfig(
        hf_path="Blablablab/multilingual-style-representation",
        model_type="sentence_transformer",
        pooling="mean",
        chunking=True,
    ),
    "all-minilm-l12-v2": ModelConfig(
        hf_path="sentence-transformers/all-MiniLM-L12-v2",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "all-minilm-l6-v2": ModelConfig(
        hf_path="sentence-transformers/all-MiniLM-L6-v2",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "all-mpnet-base-v2": ModelConfig(
        hf_path="sentence-transformers/all-mpnet-base-v2",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "all-roberta-large-v1": ModelConfig(
        hf_path="sentence-transformers/all-roberta-large-v1",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "allenai-specter": ModelConfig(
        hf_path="allenai/specter",
        model_type="encoder",
        pooling="cls",
    ),
    "bert-base-uncased": ModelConfig(
        hf_path="google-bert/bert-base-uncased",
        model_type="encoder",
        pooling="mean",
    ),
    "bge-base-en-v1.5": ModelConfig(
        hf_path="BAAI/bge-base-en-v1.5",
        model_type="sentence_transformer",
        pooling="cls",
    ),
    "bge-large-en-v1.5": ModelConfig(
        hf_path="BAAI/bge-large-en-v1.5",
        model_type="sentence_transformer",
        pooling="cls",
    ),
    "bge-small-en-v1.5": ModelConfig(
        hf_path="BAAI/bge-small-en-v1.5",
        model_type="sentence_transformer",
        pooling="cls",
    ),
    "bge-base-zh-v1.5": ModelConfig(
        hf_path="BAAI/bge-base-zh-v1.5",
        model_type="sentence_transformer",
        pooling="cls",
    ),
    "bge-large-zh-v1.5": ModelConfig(
        hf_path="BAAI/bge-large-zh-v1.5",
        model_type="sentence_transformer",
        pooling="cls",
    ),
    "bge-small-zh-v1.5": ModelConfig(
        hf_path="BAAI/bge-small-zh-v1.5",
        model_type="sentence_transformer",
        pooling="cls",
    ),
    "bge-m3": ModelConfig(
        hf_path="BAAI/bge-m3",
        model_type="sentence_transformer",
        pooling="cls",
    ),
    # E5 family
    "e5-large-v2": ModelConfig(
        hf_path="intfloat/e5-large-v2",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "e5-base-v2": ModelConfig(
        hf_path="intfloat/e5-base-v2",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "e5-small-v2": ModelConfig(
        hf_path="intfloat/e5-small-v2",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "multilingual-e5-large": ModelConfig(
        hf_path="intfloat/multilingual-e5-large",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "multilingual-e5-base": ModelConfig(
        hf_path="intfloat/multilingual-e5-base",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    # Other embedding models
    "snowflake-arctic-embed-l-v2": ModelConfig(
        hf_path="Snowflake/snowflake-arctic-embed-l-v2.0",
        model_type="sentence_transformer",
        pooling="cls",
        trust_remote_code=True,
    ),
    "jina-embeddings-v2-base-en": ModelConfig(
        hf_path="jinaai/jina-embeddings-v2-base-en",
        model_type="encoder",
        pooling="mean",
        max_length=8192,
        trust_remote_code=True,
    ),
    "jina-embeddings-v2-small-en": ModelConfig(
        hf_path="jinaai/jina-embeddings-v2-small-en",
        model_type="encoder",
        pooling="mean",
        max_length=8192,
        trust_remote_code=True,
    ),
    "mxbai-embed-large-v1": ModelConfig(
        hf_path="mixedbread-ai/mxbai-embed-large-v1",
        model_type="sentence_transformer",
        pooling="cls",
    ),
    "gte-large-en-v1.5": ModelConfig(
        hf_path="Alibaba-NLP/gte-large-en-v1.5",
        model_type="encoder",
        pooling="mean",
        trust_remote_code=True,
    ),
    "gte-base": ModelConfig(
        hf_path="thenlper/gte-base",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "gte-large": ModelConfig(
        hf_path="thenlper/gte-large",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "nv-embed-v1": ModelConfig(
        hf_path="nvidia/NV-Embed-v1",
        model_type="encoder",
        pooling="mean",
        trust_remote_code=True,
    ),
    "qwen3-embedding-0.6b": ModelConfig(
        hf_path="Qwen/Qwen3-Embedding-0.6B",
        model_type="encoder",
        pooling="mean",
        trust_remote_code=True,
    ),
    "qwen3-embedding-4b": ModelConfig(
        hf_path="Qwen/Qwen3-Embedding-4B",
        model_type="encoder",
        pooling="mean",
        trust_remote_code=True,
    ),
    "qwen3-embedding-8b": ModelConfig(
        hf_path="Qwen/Qwen3-Embedding-8B",
        model_type="encoder",
        pooling="mean",
        trust_remote_code=True,
    ),
    "instructor-xl": ModelConfig(
        hf_path="hkunlp/instructor-xl",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "instructor-large": ModelConfig(
        hf_path="hkunlp/instructor-large",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "instructor-base": ModelConfig(
        hf_path="hkunlp/instructor-base",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "facebook-contriever": ModelConfig(
        hf_path="facebook/contriever",
        model_type="encoder",
        pooling="mean",
    ),
    "facebook-contriever-msmarco": ModelConfig(
        hf_path="facebook/contriever-msmarco",
        model_type="encoder",
        pooling="mean",
    ),
    "nomic-embed-text-v1": ModelConfig(
        hf_path="nomic-ai/nomic-embed-text-v1",
        model_type="sentence_transformer",
        pooling="mean",
        trust_remote_code=True,
    ),
    "nomic-embed-text-v1.5": ModelConfig(
        hf_path="nomic-ai/nomic-embed-text-v1.5",
        model_type="sentence_transformer",
        pooling="mean",
        trust_remote_code=True,
    ),
    "paraphrase-mpnet-base-v2": ModelConfig(
        hf_path="sentence-transformers/paraphrase-mpnet-base-v2",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "paraphrase-multilingual-mpnet-base-v2": ModelConfig(
        hf_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "msmarco-distilbert-base-v4": ModelConfig(
        hf_path="sentence-transformers/msmarco-distilbert-base-v4",
        model_type="sentence_transformer",
        pooling="mean",
    ),
    "distiluse-base-multilingual-cased-v2": ModelConfig(
        hf_path="sentence-transformers/distiluse-base-multilingual-cased-v2",
        model_type="sentence_transformer",
        pooling="mean",
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    if model_name in AUTHBENCH_MODELS:
        return AUTHBENCH_MODELS[model_name]

    for name, config in AUTHBENCH_MODELS.items():
        if config.hf_path == model_name:
            return config

    raise KeyError(f"Model '{model_name}' not found in registry. Available models: {sorted(AUTHBENCH_MODELS.keys())}")
