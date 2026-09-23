import os
import sys
import logging
from typing import List, Tuple
from argparse import Namespace

import torch
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

STEER_ROOT = "/mmfs1/home/hardiksr/repository/para-summ-low-data"
if STEER_ROOT not in sys.path:
    sys.path.insert(0, STEER_ROOT)

from obfuscate.run_steer import (
    STYLES,
    style_options,
    STYLE_MAPPING,
    SPECIAL_TOKENS,
    PromptCollator,
)

class STEERGeneration:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_style = getattr(args, "target_style", "english_tweet")
        self.max_prompt_length = getattr(args, "max_prompt_length", 50)
        self.max_gen_length = getattr(args, "max_gen_length", 100)
        self.top_p = getattr(args, "top_p", 0.9)
        self.no_repeat_ngrams = getattr(args, "no_repeat_ngrams", 3)
        self.multiple_reward_tokens = getattr(args, "multiple_reward_tokens", True)
        self.n_extra_tokens = getattr(args, "n_extra_tokens", 5)
        self.num_reward_tokens = 3 if self.multiple_reward_tokens else 1
        self.nlp = spacy.load("en_core_web_sm")

        if self.multiple_reward_tokens:
            self.reward_tokens = {}
            for metric in ["style", "similarity", "fluency"]:
                self.reward_tokens[metric] = {st: [' _{}_{}_'.format(st, str(idx).zfill(5)) + metric for idx in range(self.n_extra_tokens)] 
                    for st in STYLES}
                
            reward_tokens_list = [y for x in list(self.reward_tokens["style"].values()) for y in x] + \
                                 [y for x in list(self.reward_tokens["similarity"].values()) for y in x] + \
                                 [y for x in list(self.reward_tokens["fluency"].values()) for y in x]
        else:
            self.reward_tokens = {st: [' _{}_{}'.format(st, str(idx).zfill(5)) for idx in range(self.n_extra_tokens)] 
                for st in STYLES}
            reward_tokens_list = [y for x in list(self.reward_tokens.values()) for y in x]

        base_model = getattr(args, "base_model", "gpt2-large")
        model_ckpt = getattr(args, "model_ckpt", "/gscratch/stf/kogolobo/ckp_3500.pth")

        logging.info("Loading tokenizer and configuring vocab...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, pad_token="<|endoftext|>")
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.tokenizer.add_tokens(reward_tokens_list, special_tokens=True)
        
        logging.info("Loading STEER base model config...")
        config = AutoConfig.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_config(config)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logging.info(f"Loading checkpoint {model_ckpt}...")
        checkpoint = torch.load(model_ckpt, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['policy_model'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        checkpoint.clear()
        
        self.prompt_collator = PromptCollator(tokenizer=self.tokenizer, max_prompt_length=self.max_prompt_length)
        logging.info("STEER model loaded successfully.")

    def _split_into_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _add_control_code(self, input_ids, attention_mask, target_styles):
        if not self.multiple_reward_tokens:
            best_cat_id = [self.tokenizer.convert_tokens_to_ids(self.reward_tokens[STYLE_MAPPING[t]][-1]) for t in target_styles]
            input_ids = torch.cat([input_ids.new(best_cat_id)[:, None], input_ids], dim=1)
            attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
        else:
            best_cat_ids = [[a,b,c] for a,b,c in zip(
                [self.tokenizer.convert_tokens_to_ids(self.reward_tokens["style"][STYLE_MAPPING[t]][-1]) for t in target_styles],
                [self.tokenizer.convert_tokens_to_ids(self.reward_tokens["similarity"][STYLE_MAPPING[t]][-1]) for t in target_styles],
                [self.tokenizer.convert_tokens_to_ids(self.reward_tokens["fluency"][STYLE_MAPPING[t]][-1]) for t in target_styles],
            )]
            input_ids = torch.cat([input_ids.new(best_cat_ids), input_ids], dim=1)
            attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None]] * self.num_reward_tokens + [attention_mask], dim=1)
            
        return input_ids, attention_mask

    def generate(self, prompt: str):
        sentences = self._split_into_sentences(prompt)
        
        if not sentences:
            return [prompt], {}, 0.0

        obfuscated_sentences = []
        metadata = []
        
        sequences = [{'prompt': sent, 'tgt_style': self.target_style} for sent in sentences]
        input_ids, attention_mask, tgt_styles = self.prompt_collator(sequences)
        
        with torch.no_grad():
            input_ids, attention_mask = self._add_control_code(input_ids, attention_mask, tgt_styles)
            
            generated_ids = self.model.generate(
                input_ids=input_ids.to(self.device), 
                attention_mask=attention_mask.to(self.device), 
                top_p=self.top_p, 
                max_length=self.max_gen_length, 
                min_length=2, 
                do_sample=True, 
                no_repeat_ngram_size=self.no_repeat_ngrams,
                eos_token_id=self.tokenizer.eos_token_id
            )              
            
            generations = self.tokenizer.batch_decode(
                generated_ids[:, input_ids.shape[1]:], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
        
        for i, (original_sent, gen) in enumerate(zip(sentences, generations)):
            gen = gen.strip()
            if len(gen) > 0 and gen[0].islower():
                gen = gen.capitalize()
                
            if not gen:
                gen = original_sent
                
            obfuscated_sentences.append(gen)
            metadata.append({
                "sentence": original_sent,
                "obfuscated": gen
            })
            
        final_generation = " ".join(obfuscated_sentences)

        return [final_generation], {"sentences": metadata}, 1.0
