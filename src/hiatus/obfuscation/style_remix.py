from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

class RemixRunner:
    MAX_NEW_TOKENS = 300
    def __init__(self, model_id: str, adapter_paths: Dict[str, str], lang: str = "en"):
        self.lang = lang
        self.model_id = model_id
        self.adapter_paths = adapter_paths
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, add_bos_token=True, add_eos_token=False, padding_side="left")
        self.tokenizer.add_special_tokens({'pad_token': '<padding_token>'})
        base_model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        base_model.resize_token_embeddings(len(self.tokenizer)) 
        
        # Load in the first LoRA adapter
        first_model = list(adapter_paths.keys())[0]
        self.model = PeftModel.from_pretrained(base_model, adapter_paths[first_model], adapter_name=first_model).to(self.device)
        if not self.model.generation_config.pad_token_id:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        # Load in the rest of the models
        for cur_adapter in tqdm(adapter_paths.keys(), desc="Loading adapters"):
            if cur_adapter != first_model:
                self.model.load_adapter(adapter_paths[cur_adapter], adapter_name=cur_adapter)
        self.model.to(self.device)
        self.model.eval()
        self.curr_adapter_name = None
        
    def remix(self, text: str, directions: Dict[str, float]) -> str:
        directions = self.normalize_directions(directions)
        self.replace_adapters(directions)
        prompt = self.craft_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(self.device)
        input_length = inputs.input_ids.shape[1]

        decoding_params = {"top_p": 0.95, "max_new_tokens": self.MAX_NEW_TOKENS}
        if self.lang == "ru":
            decoding_params.update({
                "top_k": 40,
                "repetition_penalty": 1.5,
                "do_sample": True,
                "temperature": 0.3,
            })
        with torch.no_grad(): 
            outputs = self.model.generate(**inputs, **decoding_params)
        response = self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
        return response
        
    def normalize_directions(self, directions: Dict[str, float]):
        norm_directions = {}
        for direction, weight in directions.items():
            if weight is not None and weight != 0:    
                if 'type' in direction:
                    norm_directions[direction] = abs(weight)
                elif weight < 0:
                    direction = direction.replace("more", "less")
                    norm_directions[direction] = abs(weight)
                else:
                    norm_directions[direction] = weight
                    
        return norm_directions
    
    def replace_adapters(self, directions: Dict[str, float]):
        adapter_name = '-'.join([f"{direction}_{100*weight:.0f}" for direction, weight in directions.items()])
        if adapter_name != self.curr_adapter_name:
            self.model.add_weighted_adapter(
                list(directions.keys()),
                weights = list(directions.values()),
                adapter_name = adapter_name,
                combination_type = "cat"
            )
            self.model.set_adapter(adapter_name)
            if self.curr_adapter_name is not None:
                self.model.delete_adapter(self.curr_adapter_name)
                
        else:
            self.model.set_adapter(adapter_name)
        
        self.curr_adapter_name = adapter_name
        
    def craft_prompt(self, text: str):
        if self.lang == "ru":
            return f"### Оригинал: {text}\n ### Перепишите: "
        elif self.lang == "en":
            return f"### Original: {text}\n ### Rewrite:"
                