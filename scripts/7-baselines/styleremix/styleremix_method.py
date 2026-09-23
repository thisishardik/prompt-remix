import os
import sys
import torch
import numpy as np
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

STYLE_REMIX_ROOT = "/mmfs1/home/hardiksr/repository/StyleRemix"
if STYLE_REMIX_ROOT not in sys.path:
    sys.path.insert(0, STYLE_REMIX_ROOT)

from quickstart import MODEL_PATHS, convert_data_to_format

class StyleRemixGeneration:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = getattr(args, "model_ckpt", "meta-llama/Meta-Llama-3-8B")
        self.hf_token = getattr(args, "hf_token", None)
        self.max_new_tokens = getattr(args, "max_new_tokens", 1024)
        
        print(f"Loading base model {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            add_bos_token=True, 
            add_eos_token=False, 
            padding_side="left",
            token=self.hf_token
        )
        self.tokenizer.add_special_tokens({'pad_token': '<padding_token>'})
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            token=self.hf_token,
            device_map="auto" if torch.cuda.is_available() else None
        )
        base_model.resize_token_embeddings(len(self.tokenizer))
        
        first_model = list(MODEL_PATHS.keys())[0]
        print(f"Loading primary adapter {first_model}...")
        self.model = PeftModel.from_pretrained(
            base_model, 
            MODEL_PATHS[first_model], 
            adapter_name=first_model,
            token=self.hf_token
        ).to(self.device)
        
        for cur_adapter in MODEL_PATHS.keys():
            if cur_adapter != first_model:
                print(f"Loading adapter {cur_adapter}...")
                self.model.load_adapter(MODEL_PATHS[cur_adapter], adapter_name=cur_adapter, token=self.hf_token)
                
        self.model.eval()
        print("StyleRemix model and adapters loaded successfully.")
        self.curr_adapter_name = None

    def _set_adapter(self, weights_dict: Dict[str, float]):
        sliders_dict = {}
        for key, weight in weights_dict.items():
            if weight != 0:
                sliders_dict[key] = abs(weight)
                
        if len(sliders_dict) > 0:
            combo_adapter_name = "-".join([f"{k}{int(100*v)}" for k, v in sliders_dict.items()])
            
            if combo_adapter_name != self.curr_adapter_name:
                self.model.add_weighted_adapter(
                    list(sliders_dict.keys()),
                    weights=list(sliders_dict.values()),
                    adapter_name=combo_adapter_name,
                    combination_type="cat"
                )
                self.model.set_adapter(combo_adapter_name)
                
                if self.curr_adapter_name is not None:
                    self.model.delete_adapter(self.curr_adapter_name)
                    
            else:
                self.model.set_adapter(combo_adapter_name)
                
            self.curr_adapter_name = combo_adapter_name
            return True
        return False

    def generate(self, prompt: str, author_weights: Dict[str, float]) -> Tuple[List[str], Dict, float]:
        has_adapters = self._set_adapter(author_weights)
        
        if not has_adapters:
            return [prompt], {}, 0.0

        converted_text = convert_data_to_format(prompt)
        inputs = self.tokenizer(converted_text, return_tensors="pt", max_length=2048, truncation=True).to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, top_p=0.95)
            
        response = self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
        
        return [response], {"author_weights": author_weights}, 1.0
