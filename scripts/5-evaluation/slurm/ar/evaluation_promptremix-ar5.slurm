#!/bin/bash

#SBATCH --job-name=prompt_remix_evaluation-ar5
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=BEGIN,FAIL,END

#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00 

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/evaluation_promptremix-ar5.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/evaluation_promptremix-ar5.err

source /mmfs1/home/hardiksr/.bashrc
conda activate hiatus
module load cuda gcc

# LLM as Evaluator (see 5-evaluation/llm_as_evaluator.py for list of criteria)
original_text_path="/gscratch/stf/hardiksr/hiatus/data/ar/evaluation/queries/hrs_06-27-24_arabic_crossGenre-combined_TA3_input_queries.jsonl"
privatized_text_path="/gscratch/stf/hardiksr/hiatus/data/ar/obfuscation/hrs_06-27-24_arabic_crossGenre-combined_TA3_remix-ar5.jsonl"
output_path_llm="/gscratch/stf/hardiksr/hiatus/metrics/ar/promptremix-TA3_ar5_llm.jsonl"
output_path_perplexity="/gscratch/stf/hardiksr/hiatus/metrics/ar/perplexity_metrics.jsonl"

# Sense Scores
python scripts/5-evaluation/score_sense.py \
    --text_original_jsonl $original_text_path \
    --text_privatized_jsonl $privatized_text_path \
    --output_path $output_path_llm \
    --model_path "mistralai/Mistral-Small-24B-Instruct-2501"
    #--model_path "mistralai/Mistral-Small-3.2-24B-Instruct-2506"


# Perplexity Score (not including first word of the sentence due to absence of BOS tokens in the used BERT model)
# python 5-evaluation/get_perplexity.py \
#    --input_path $privatized_text_path \
#    --ref_key "fullText" \
#    --output_path $output_path_perplexity  \
#    --model "hfl/chinese-roberta-wwm-ext-large" \
#    --max_length 512 