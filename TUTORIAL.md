## Tutorial

This project is a multistep effort to accomplish the author obfuscation goal using ideas from the PromptRemix method. All terms in this documentation refer to their pertinent meaning within the scope of StyleRemix/PromptRemix (e.g., obfuscation has a more specific meaning when using PromptRemix than it has when describing authorship obfuscation as a general task). For a description of PromptRemix and StyleRemix, see the [Glossary](#glossary).

### 1. Pre-obfuscation

Consists in creating a training set for each style axis to be modified.
1. **Data collection**: collection of texts from a web resource (e.g., Wikipedia) to train style classifiers in a standardized format (e.g., paragraphs with at most `N_CHAR` characters). See `scripts/1-data/`.
2. **Prompt engineering**: construction of prompts to be fed to LLMs to generate texts with increased/decreased levels of a style axis (e.g., formality). See `prompts/`.
3. **Paraphrasing**: generation of texts using LLMs using prompts from Step 2. See `scripts/2-paraphrase`.
4. **Classification**: training of a classifier model (e.g., BERT model with classifier head using `AutoModelForSequenceClassification` on HF) on the paraphrased texts to predict increased/decreased levels of the style axis. See `scripts/3-classifiers`.

### 2. Obfuscation

Consists in identifying, for a given author, its prominent stylistic features (i.e., the author's invariants) and perturbing the text along the corresponding invariant axes (aka, style axes) to bring their values closer to the average on the population. 

1. **PII injection**: addition of natural, gender-neutral first names (e.g., Alex in en, Sasha in ru, 晨 in zh) as mocks to replace `<PERSON>` tags in evaluation data. See `scripts/4-obfuscation/insert_pii.py`
2. **Remix**: the main obfuscation step. See the [Glossary](#glossary) for a high-level explanation of StyleRemix and PromptRemix. See `4-obfuscation/run_remix.py` for the execution script and `src/hiatus/obfuscation/{prompt,style}_remix.py` for the source code.
3. **PII removal**: removal of mock first names from obfuscated text. See `scripts/4-obfuscation/remove_pii.py`.

### 3. Evaluation

Consists in evaluating the obfuscated text according to several metrics, such as linguistic acceptability (with CoLA models), coherence, fluency, quality and understandability (these latter being computed by prompting a large multilingual pre-trained LLM).

1. **LLM-based Evaluation**: metrics
    * Mistral model: Mistral models are pre-trained multilingual models used for sense scores. See `5-eveluation/llm_as_evaluator.py`.
    * Cosine Similarity for Sentence Embeddings: see `src/hiatus/evaluation/meaning_sim_contextual_embeddings.py`.
    * Perplexity: see `src/hiatus/evaluation/perplexity.py`.
    * CoLA: as of August 2025, there are no CoLA models for Chinese. See `5-evaluation/get_scores.py` for a script that can be used with a CoLA model.  
2. **Privacy Evaluation**: based on models from other technical areas (TA), specifically models from TA1 and TA2 developed by other groups involved in HIATUS. Code will be eventually be shared by teammates.

### 4. Report

Consists of methods for producing visual displays of metrics (e.g., tables and graphs) in the evaluation round. These displays are then used to prepare presentations in the project meetings. Meetings happen every 2 weeks, typically at 10 a.m. You are expected to prepare short presentation (2-4 slides, divided roughly in 1 methodology slide, 1 metrics slides, 1 slide explaining experiments and a results slide with visual/textual aids).
1. **Metrics**: all the metrics from [Evaluation](#3-evaluation).
2. **Visual Aids**: bar charts with pastel colors (see, for instance, the bar plots used in comparing LLMs in benchmarks).
3. **Textual Aids**: tables with metrics.
4. **Communication**: be flexible, but not entirely open, to other's contributions. It is important to stick to your point and only make eventual concessions about future work. Every concession is a commitment to produce something for the next meeting.

### Glossary
* Low-rank Adaptation (LoRA): LoRA is a technique for fine-tuning LLMs introduced in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). It works by freezing weights of a large pre-trained model and instead fitting only trainable rank decompositions in the layers of the Transformer architecture. This greatly reduces the number of trainable parameters for fine-tuning to downstream tasks.
* StyleRemix: is an author obfuscation method based on [StyleRemix: Interpretable Authorship Obfuscation via Distillation and Perturbation of Style Elements](https://arxiv.org/abs/2408.15666). In StyleRemix, the perturbation across the style axis is performed by prompting LoRA adapters (fine-tuned on the same dataset as the classifiers, typically) to rewrite the story across the style axis it is adapted to. See `src/hiatus/obfuscation/style_remix.py`.
* PromptRemix: is an author obfuscation method based on StyleRemix. The core distinction between PromptRemix and StyleRemix is in the perturbation step. In PromptRemix, the perturbation is performed by prompting a LLM (e.g., Qwen, Llamma) with instructions on how to perturb the text along a style axis and how much the text should be perturbed (in a qualitative scale of "weak", "medium", "strong", "very strong"). See `src/hiatus/obfuscation/prompt_remix.py`.

## Future work

1. Backward compatibility with pipeline for Russian and English:
    * Changes introduced in how the type/voice style axis is handled and how `remix_config.json` is constructed are not backward compatible. Consider modifying the Russian and English pipeline.
2. Extension to Arabic: this requires changes across the library, for instance,
    * Natural Language Processing (NLP) support for Arabic: as of August 2025, `spaCy` does not support Arabic. Consider using `stanza` for Arabic. 
    * BERT-like models for Arabic: consider checking the models in [CAMeL-Lab](https://huggingface.co/CAMeL-Lab) developed in NYU Abu Dhabi.
3. Defining baselines:
    * Translate-backtranslate (zh → en → zh).
4. Adapting the pipeline to support easy training of baseline methods. Should involve `3-classifiers` (for the translate-backtranslate model), `4-obfuscation` (for obfuscating the translate language), `5-evaluation` and `6-report`.

