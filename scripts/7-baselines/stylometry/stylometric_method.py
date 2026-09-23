"""This file support main_evaluation.py
"""
from typing import Union, List
import torch
import spacy
from typing import List
import torch
import numpy as np
from utils_stylo import find_cosine_similarity, average_repeated_values, find_sentence_attribute, set_seed, evaluation_quality
from transformers import (T5ForConditionalGeneration, T5TokenizerFast, AutoModelForSequenceClassification, AutoTokenizer)
from sentence_transformers import SentenceTransformer


class StylometricGeneration(): 
    STOP_TOKEN = "<|endoftext|>"

    def __init__(
        self, 
        args,
        seed: int = 42,
        device: str = 'cuda:0',
        word_embedding_dict_dir: str = "/mmfs1/home/hardiksr/repository/JAMDecoding/dataset/top_20K_T5_word_embeddings"
    ):
        self.device = device
        n_gpu = torch.cuda.device_count()
        set_seed(seed, n_gpu)
        self.end_of_text_token = "<|endoftext|>"
        self.start_of_text_token = "<|startoftext|>"
        self.seperator_token = "<|seperator|>"
        self.pad_token = "<|pad|>"
        self.tokenizer =  T5TokenizerFast.from_pretrained('t5-base', bos_token='<|startoftext|>',                                                       
                                                        sep_token= "<|seperator|>", cache_dir = args.cache_dir) 
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base', cache_dir = args.cache_dir).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.T5_word_embeddings_dict = torch.load(word_embedding_dict_dir)
        T5_word_embeddings = []
        for t in self.T5_word_embeddings_dict.keys():
            T5_word_embeddings.append(self.T5_word_embeddings_dict[t])
        T5_word_embeddings = torch.stack(T5_word_embeddings)
        self.T5_word_embeddings = T5_word_embeddings.reshape(T5_word_embeddings.shape[0],T5_word_embeddings.shape[2])

        self.eval_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA', cache_dir = args.cache_dir).to(device)
        self.eval_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA', cache_dir = args.cache_dir)
        self.sent_model = SentenceTransformer('sentence-transformers/sentence-t5-base', cache_folder=args.cache_dir)

    def __repr__(self):
        return f'<StylometricGenerator model_name_or_path="{self.model}">'

    def generate(self,
                 prompt: Union[str, List[str]],
                 max_len: int = 20,
                 sample: bool = True,
                 k: int = 5,
                 temperature: float = 1.0,
                 alpha: float = 1.0,
                 cola_threshold: float = 0.0,
                 sentence_similarity = True, 
                 avoid_ls = None,
                 avoid_weight = 1.0 ,
                 verbose_threshold = 1.0):
        if isinstance(prompt, str):
            prompt = [prompt]

        nlp = spacy.load("en_core_web_sm")
        spacy_original_prompt = nlp(prompt[0])

        if avoid_ls != None:
            avoid_ls_lemma = [nlp(a)[0].lemma_ for a in avoid_ls]

        candidate_data = {'prompt': prompt}    
        with torch.no_grad():
            for step in range(len(spacy_original_prompt)):
                if str(spacy_original_prompt[step]) == '\n':
                    continue
                pos = spacy_original_prompt[step].pos_
                if pos in ["VERB", "NOUN", "ADJ"] and str(spacy_original_prompt[step]) in self.T5_word_embeddings_dict.keys():
                    if step == 0:
                        input_ids = None
                    if pos in ["ADJ"]:
                        if verbose_threshold != 1.0:
                            cola_difference = find_sentence_attribute(spacy_original_prompt, step, self.sent_model, self.tokenizer, input_ids, [""], self.device, self.eval_model, self.eval_tokenizer, fill_in = True, metric = "cola_difference" )
                            if cola_difference < verbose_threshold:
                                continue
                    original_word = spacy_original_prompt[step]
                    original_word_embedding = self.T5_word_embeddings_dict[str(original_word)]

                    similarity_scores = find_cosine_similarity(original_word_embedding, self.T5_word_embeddings)
                    candidates = list(self.T5_word_embeddings_dict.keys())
                    original_raw = str(original_word).strip().lower()
                    vals_norepeat, idx_norepeat, candidate_words = average_repeated_values(candidates, similarity_scores, k, self.tokenizer, original_raw, spacy_original_prompt[step], self.device)
                    if sentence_similarity:
                        vals_norepeat = find_sentence_attribute(spacy_original_prompt, step, self.sent_model, self.tokenizer, input_ids, candidate_words, self.device, self.eval_model, self.eval_tokenizer, fill_in = True, metric = "cosine_similarity").to(self.device)
                    if avoid_ls != None:
                        for i, candidate in enumerate([nlp(c)[0] for c in candidate_words]):
                            if candidate.lemma_ in avoid_ls_lemma:
                                vals_norepeat[0][i]=torch.tensor(avoid_weight).to(self.device)
                    val_min = torch.min(vals_norepeat)
                    val_max = torch.max(vals_norepeat)
                    if val_max == val_min:
                        standardize_vals_norepeat = [torch.ones_like(i) for i in vals_norepeat]
                    else:
                        standardize_vals_norepeat = [(i - val_min)/(val_max - val_min) for i in vals_norepeat]

                    if step == 0:
                        input_ids = None
                    cola_average = find_sentence_attribute(spacy_original_prompt, step, self.sent_model, self.tokenizer, input_ids, candidate_words, self.device, self.eval_model, self.eval_tokenizer, fill_in = True, metric = "cola")
                    if cola_threshold != 1.0:
                        cola_average = [c if c>cola_threshold else 0 for c in cola_average]
                    if len([c for c in cola_average if c==0]) == len(candidate_words):
                        standardize_cola_averages = torch.zeros(len(candidate_words)).to(self.device)
                    else:
                        cola_min = np.min(cola_average)
                        cola_max = np.max(cola_average)
                        if cola_max == cola_min:
                            standardize_cola_averages = torch.ones(len(candidate_words)).to(self.device)
                        else:
                            standardize_cola_averages = torch.tensor([(float(i) - cola_min)/(cola_max - cola_min) for i in cola_average]).to(self.device)
                    
                    candidate_key = str(step)+"_"+str(original_word)
                    candidate_data[candidate_key] = {"original_word": str(original_word), 
                                                    "similarity_core": vals_norepeat,
                                                    "similar_words": candidate_words,
                                                    "similar_tokens": idx_norepeat,
                                                    "cola_average":cola_average
                                                     }
                    combine_vals = standardize_vals_norepeat[0] + alpha * standardize_cola_averages
                    combine_vals = combine_vals.reshape((1,len(combine_vals))).float()
                    next_token_logits = combine_vals 

                    if sample:
                        if temperature != 1.0:
                            next_token_logits = next_token_logits / temperature
                        probs = next_token_logits
                        probs = torch.clamp(probs, min=0.0)
                        if probs.sum() == 0:
                            probs = torch.ones_like(probs)
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        next_tokens = torch.argmax(next_token_logits, dim=-1)

                    next_tokens = idx_norepeat[next_tokens.item()]
                else:
                    next_word = str(spacy_original_prompt[step])
                    next_token = self.tokenizer(next_word)['input_ids']

                    if 1 in next_token:
                        next_token.remove(1)
                    next_tokens = torch.tensor(next_token).to(self.device)
                tokens_to_add = torch.cat((next_tokens, torch.tensor([self.tokenizer.pad_token_id ]).to(self.device))).reshape([1,next_tokens.shape[0]+1])
                if step == 0:
                    input_ids = tokens_to_add
                else:
                    input_ids = torch.cat([input_ids, tokens_to_add], dim=-1)
        decoded_outputs = [self.tokenizer.decode(output.int(), clean_up_tokenization_spaces=True)
                           for output in input_ids]
                           
        words = []
        for i, word in enumerate(decoded_outputs[0].split("<pad>")):
            if i == 0:
                words.append(word)
            elif word == '':
                continue
            elif word in [" .", " !", " ?", " ,", " ;", " :", " )", " ]"]:
                words.append(word[1:])
            elif word in [".", "!", "?"]:
                words.append(word)
            elif word[0] != " ":
                words.append(" " + word)
            else:
                words.append(word)
        final_generation = "".join(words)
        final_generation_cola = evaluation_quality([final_generation], self.device, self.eval_model, self.eval_tokenizer)
        return [final_generation], candidate_data, final_generation_cola[1]
