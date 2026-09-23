import torch
import spacy
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

class ParaphraseGeneration:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nlp = spacy.load("en_core_web_sm")

        model_name = 'tuner007/pegasus_paraphrase'
        cache_dir = getattr(args, 'cache_dir', None)
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)

    def generate(self, prompt: str, num_return_sequences: int = 1, num_beams: int = 10, temperature: float = 1.5):
        doc = self.nlp(prompt)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            return [prompt], {}, 0.0
            
        obfuscated_sentences = []
        
        with torch.no_grad():
            for sent in sentences:
                batch = self.tokenizer([sent], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(self.device)
                translated = self.model.generate(
                    **batch,
                    max_length=60,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature
                )
                tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
                obfuscated_sentences.append(tgt_text[0])

        final_generation = " ".join(obfuscated_sentences)
        return [final_generation], {}, 1.0
