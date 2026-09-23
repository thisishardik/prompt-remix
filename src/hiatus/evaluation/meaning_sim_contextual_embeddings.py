import logging
import os
import sys
from argparse import Namespace
from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout)


class MultiLingualSentenceMeaningSimilarityMetric:
    """
    A class to evaluate the semantic similarity between the original and obfuscated documents.

    Attributes:
        args (Namespace): Configuration parameters including device and batch size.
        device (torch.device): The device (CPU or GPU) to perform computations.
        length_threshold (float): Threshold to determine text length impact.
        batch_size (int): Number of documents processed in one batch.
        tokenizer (AutoTokenizer): Pre-trained tokenizer for encoding texts.
        model (AutoModel): Pre-trained model for generating text embeddings.
    """

    def __init__(self, args: Namespace):
        self.args = args
        self.device = args.device
        self.length_threshold = 1.5
        self.batch_size = args.batch_size
        model_name="intfloat/multilingual-e5-large"
        # model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # model_name = "/nfs/nimble/projects/hiatus/models/intfloat/multilingual-e5-large"  # multi-lingual model support both english and russian
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)

    # Mean Pooling - Take attention mask into account for correct averaging
    # def mean_pooling(self, model_output, attention_mask):
    #     token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def average_pool(self, last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, texts):
        # Process texts in batches
        all_encodings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            sentence_embeddings = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            all_encodings.extend(sentence_embeddings)

        return torch.stack(all_encodings)

    def infer_meaning_similarity(self, orig_docs, obf_docs):
        meaning_similarity_values = {}

        # Extract and encode all texts
        orig_texts = []
        obf_texts = []
        doc_ids = []
        for doc_id, doc in orig_docs.items():
            orig_texts.append(doc["fullTextOriginal"])
            obf_texts.append(obf_docs[doc_id]["fullTextOriginal"])
            doc_ids.append(doc_id)

        orig_encodings = self.encode(orig_texts)
        obf_encodings = self.encode(obf_texts)

        # Calculate cosine similarity for each pair
        for doc_id, orig_enc, obf_enc in zip(doc_ids, orig_encodings, obf_encodings):
            similarity_score = cosine_similarity(orig_enc.cpu().unsqueeze(0), obf_enc.cpu().unsqueeze(0))
            transformed_similarity_score = (similarity_score + 1) / 2
            # meaning_similarity_values.append((doc_id, transformed_similarity_score.item()))
            meaning_similarity_values[doc_id] = transformed_similarity_score.item()

        return meaning_similarity_values

    def infer_meaning_similarity_sent_level(self, orig_sentences, obf_sentences):
        doc_similarities = []
        sentence_level_similarity_values = {}
        for sent_idx, sent in orig_sentences.items():
            # sent_idx = int(sent_idx)
            #     return {i: {"text": sent.text.strip()} for i, sent in enumerate(doc.sents)}
            orig_sent = sent['text']
            obf_sent = obf_sentences[sent_idx]['text']
            orig_enc = self.encode([orig_sent])
            obf_enc = self.encode([obf_sent])
            similarity_score = cosine_similarity(orig_enc.cpu(), obf_enc.cpu())
            transformed_similarity_score = (similarity_score + 1) / 2
            doc_similarities.append(transformed_similarity_score.item())
            sentence_level_similarity_values[int(sent_idx)] = transformed_similarity_score.item()

        return sentence_level_similarity_values


class SentenceMeaningSimilarityMetric:
    """
    A class to evaluate the semantic similarity between the original and obfuscated documents.

    Attributes:
        args (Namespace): Configuration parameters including device and batch size.
        device (torch.device): The device (CPU or GPU) to perform computations.
        length_threshold (float): Threshold to determine text length impact.
        batch_size (int): Number of documents processed in one batch.
        tokenizer (AutoTokenizer): Pre-trained tokenizer for encoding texts.
        model (AutoModel): Pre-trained model for generating text embeddings.
    """

    def __init__(self, args: Namespace):
        self.args = args
        self.device = args.device
        self.length_threshold = 1.5
        self.batch_size = args.batch_size
        # model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # multi-lingual model support both english and russian
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir,
                                                       local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=args.cache_dir,
                                               local_files_only=True)
        self.model = self.model.to(self.device)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts):
        # Process texts in batches
        all_encodings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Perform pooling
            sentence_embeddings = self.mean_pooling(outputs, inputs['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            all_encodings.extend(sentence_embeddings)

        return torch.stack(all_encodings)

    def infer_meaning_similarity(self, orig_docs, obf_docs):
        meaning_similarity_values = {}

        # Extract and encode all texts
        orig_texts = []
        obf_texts = []
        doc_ids = []
        for doc_id, doc in orig_docs.items():
            orig_texts.append(doc["fullTextOriginal"])
            obf_texts.append(obf_docs[doc_id]["fullTextOriginal"])
            doc_ids.append(doc_id)

        orig_encodings = self.encode(orig_texts)
        obf_encodings = self.encode(obf_texts)

        # Calculate cosine similarity for each pair
        for doc_id, orig_enc, obf_enc in zip(doc_ids, orig_encodings, obf_encodings):
            similarity_score = cosine_similarity(orig_enc.cpu().unsqueeze(0), obf_enc.cpu().unsqueeze(0))
            transformed_similarity_score = (similarity_score + 1) / 2
            # meaning_similarity_values.append((doc_id, transformed_similarity_score.item()))
            meaning_similarity_values[doc_id] = transformed_similarity_score.item()

        return meaning_similarity_values

    def infer_meaning_similarity_sent_level(self, orig_sentences, obf_sentences):
        doc_similarities = []
        sentence_level_similarity_values = {}
        for sent_idx, sent in orig_sentences.items():
            # sent_idx = int(sent_idx)
            #     return {i: {"text": sent.text.strip()} for i, sent in enumerate(doc.sents)}
            orig_sent = sent['text']
            obf_sent = obf_sentences[sent_idx]['text']
            orig_enc = self.encode([orig_sent])
            obf_enc = self.encode([obf_sent])
            similarity_score = cosine_similarity(orig_enc.cpu(), obf_enc.cpu())
            transformed_similarity_score = (similarity_score + 1) / 2
            doc_similarities.append(transformed_similarity_score.item())
            sentence_level_similarity_values[int(sent_idx)] = transformed_similarity_score.item()

        return sentence_level_similarity_values


class DocumentMeaningSimilarityMetric:
    """
    A class to verify the semantic consistency between the original and obfuscated documents.
    """

    def __init__(self, args: Namespace):
        self.args = args
        self.device = args.device
        self.batch_size = args.batch_size

        # Load the Stella model
        self.model_path = "/nfs/nimble/projects/hiatus/models/dunzhang/stella_en_1.5B_v5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True,
                                                       cache_dir=args.cache_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, cache_dir=args.cache_dir,
                                               local_files_only=True).to(self.device).eval()
        print(f"Loaded model from {self.model_path}")

        # Load the vector linear layer (from Stella's fine-tuning path)
        vector_dim = 1024
        vector_linear_directory = f"2_Dense_{vector_dim}"
        self.vector_linear = torch.nn.Linear(self.model.config.hidden_size, vector_dim)
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(self.model_path, f"{vector_linear_directory}/pytorch_model.bin")).items()
        }
        self.vector_linear.load_state_dict(vector_linear_dict)
        self.vector_linear = self.vector_linear.to(self.device)

        # Query prompt template
        # self.query_prompt_template = (
        #     "Instruct: You are to verify that an obfuscated text faithfully retains the meaning of an original text "
        #     "without introducing irrelevant content, contradictions, hallucinations, or omitting essential details. "
        #     "Carefully compare both versions to ensure semantic equivalence and factual consistency. \n"
        #     "Original Text:"
        # )
        # Query prompt template
        # self.query_prompt_template = (
        #     "Instruct: The following text is the original document. Your goal is to produce an embedding "
        #     "that captures its complete meaning, factual details, and context. This embedding will be used to check "
        #     "if any obfuscated version retains the full semantic integrity without introducing irrelevant content, contradictions, "
        #     "hallucinations, or omitting critical information. Focus on representing the original text's key facts, "
        #     "intent, and nuances so that any discrepancies in an obfuscated version can be accurately detected.\n"
        #     "Original Document:"
        #
        # )

        self.query_prompt_template = (
            "Instruct: Given a query document, retrieve relevant rephrased documents that retains complete semantic "
            "integrity, factual details, and context including all key facts, intent, and nuances without introducing "
            "irrelevant content, contradictions, hallucinations, or omitting critical information\n"
            "Query Document: "
        )

    def embed_texts(self, texts: List[str], is_query: bool = False):
        """
        Embed texts (queries or documents) into dense vectors using Stella.
        """
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Add prompts for queries if needed
            if is_query:
                batch_texts = [self.query_prompt_template + text for text in batch_texts]

            with torch.no_grad():
                inputs = self.tokenizer(batch_texts, padding="longest", truncation=True, max_length=4096,
                                        return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                attention_mask = inputs["attention_mask"]
                last_hidden_state = self.model(**inputs)[0]
                last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)

                # Mean pooling
                vector = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                vector = normalize(self.vector_linear(vector).cpu().numpy())

                embeddings.extend(vector)

        return embeddings

    def infer_meaning_similarity(self, orig_docs, obf_docs):
        """
        Compare original documents with obfuscated documents to assess meaning preservation.
        """
        meaning_similarity_values = {}

        # Extract and encode all texts
        orig_texts = []
        obf_texts = []
        doc_ids = []
        for doc_id, doc in orig_docs.items():
            orig_texts.append(doc["fullTextOriginal"])
            obf_texts.append(obf_docs[doc_id]["fullTextOriginal"])
            doc_ids.append(doc_id)

        orig_embeddings = self.embed_texts(orig_texts, is_query=True)
        obf_embeddings = self.embed_texts(obf_texts, is_query=False)

        # Compute cosine similarity
        for doc_id, orig_embedding, obf_embedding in zip(doc_ids, orig_embeddings, obf_embeddings):
            similarity_score = cosine_similarity(orig_embedding.reshape(1, -1), obf_embedding.reshape(1, -1))[0][0]
            transformed_similarity_score = (similarity_score + 1) / 2  # Scale to (0, 1)
            meaning_similarity_values[doc_id] = transformed_similarity_score.item()

        return meaning_similarity_values


class MultiLingualDocumentMeaningSimilarityMetric:
    """
    A class to verify the semantic consistency between the original and obfuscated documents.
    """

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def normalize(self, embeddings: Tensor, p=2, dim=1) -> Tensor:
        return F.normalize(embeddings, p=p, dim=dim)

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16

        # Load the E5 Mistral model
        self.model_path = "intfloat/e5-mistral-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device).eval()
        print(f"Loaded model from {self.model_path}")
        # Query prompt template
        # self.query_prompt_template = (
        #     "Instruct: You are to verify that an obfuscated candiate document faithfully retains the meaning of an query document "
        #     "without introducing contradictions, hallucinations, or omitting essential details. "
        #     "Carefully compare both versions to ensure semantic equivalence and factual consistency.\n"
        #     "Original Document:"
        # )
        # self.query_prompt_template = (
        #     "Instruct: The following text is the original document. Your goal is to produce an embedding "
        #     "that captures its complete meaning, factual details, and context. This embedding will be used to check "
        #     "if any obfuscated version retains the full semantic integrity without introducing irrelevant content, contradictions, "
        #     "hallucinations, or omitting critical information. Focus on representing the original text's key facts, "
        #     "intent, and nuances so that any discrepancies in an obfuscated version can be accurately detected.\n"
        #     "Original Document: "
        # )
        self.query_prompt_template = (
            "Instruct: Given a query document, retrieve relevant rephrased documents that retains complete semantic "
            "integrity, factual details, and context including all key facts, intent, and nuances without introducing "
            "irrelevant content, contradictions, hallucinations, or omitting critical information\n"
            "Query Document: "
        )

    def embed_texts(self, texts: List[str], is_query: bool = False):
        """
        Embed texts (queries or documents) into dense vectors using the E5 model.
        """
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Add prompts for queries if needed
            if is_query:
                batch_texts = [self.query_prompt_template + text for text in batch_texts]

            with torch.no_grad():
                inputs = self.tokenizer(batch_texts, max_length=4096, padding=True, truncation=True,
                                        return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                embeddings_batch = self.last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                embeddings_batch = self.normalize(embeddings_batch)

                embeddings.extend(embeddings_batch.cpu().numpy())

        return embeddings

    def infer_meaning_similarity(self, orig_docs, obf_docs):
        """
        Compare original documents with obfuscated documents to assess meaning preservation.
        """
        meaning_similarity_values = {}

        # Extract and encode all texts
        orig_texts = []
        obf_texts = []
        doc_ids = []
        for doc_id, doc in orig_docs.items():
            orig_texts.append(doc["fullText"])
            obf_texts.append(obf_docs[doc_id]["fullText"])
            doc_ids.append(doc_id)

        orig_embeddings = self.embed_texts(orig_texts, is_query=True)
        obf_embeddings = self.embed_texts(obf_texts, is_query=False)

        # Compute cosine similarity
        for doc_id, orig_embedding, obf_embedding in zip(doc_ids, orig_embeddings, obf_embeddings):
            similarity_score = cosine_similarity(orig_embedding.reshape(1, -1), obf_embedding.reshape(1, -1))[0][0]
            transformed_similarity_score = (similarity_score + 1) / 2  # Scale to (0, 1)
            meaning_similarity_values[doc_id] = transformed_similarity_score

        return meaning_similarity_values
    
    def unload_model(self):
        del self.model
        self.model = None
        del self.tokenizer
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
