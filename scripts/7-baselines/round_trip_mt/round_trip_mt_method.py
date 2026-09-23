import torch
import spacy
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class RoundTripMTGeneration:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nlp = spacy.load("en_core_web_sm")

        model_name = 'facebook/m2m100_418M'
        cache_dir = getattr(args, 'cache_dir', None)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)

        self.lang_id_en = self.tokenizer.get_lang_id("en")
        self.lang_id_de = self.tokenizer.get_lang_id("de")
        self.lang_id_fr = self.tokenizer.get_lang_id("fr")

    def _translate(self, texts, src_lang_code, tgt_lang_id):
        self.tokenizer.src_lang = src_lang_code
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        generated_tokens = self.model.generate(
            **encoded, 
            forced_bos_token_id=tgt_lang_id
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def generate(self, prompt: str):
        doc = self.nlp(prompt)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            return [prompt], {}, 0.0
            
        obfuscated_sentences = []
        
        with torch.no_grad():
            for sent in sentences:
                de_text = self._translate([sent], "en", self.lang_id_de)[0]
                fr_text = self._translate([de_text], "de", self.lang_id_fr)[0]
                en_text = self._translate([fr_text], "fr", self.lang_id_en)[0]
                obfuscated_sentences.append(en_text)

        final_generation = " ".join(obfuscated_sentences)
        return [final_generation], {}, 1.0
