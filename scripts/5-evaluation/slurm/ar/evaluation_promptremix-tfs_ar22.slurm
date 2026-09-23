#!/bin/bash

#SBATCH --job-name=evaluation_promptremix-tfs_ar22
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=120GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=50:00:00 

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/evaluation_promptremix_ar22.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/evaluation_promptremix_ar22.err
set -e

source /mmfs1/home/hardiksr/.bashrc
module load cuda gcc

# Number of styles to update
num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}

# Genres for Arabic
ar_genres=("ar_twitter_1")

for num_styles in "${num_styles_to_change[@]}"; do
    pipeline_suffix="promptremix-tfs_ar22-${num_styles}"

    for genre in "${ar_genres[@]}"; do
        echo "Running evaluation for genre: ${genre}"

        # --- Privacy and Stylistic Consistency ---
        conda activate hiatus-eval
        python scripts/5-evaluation/score_privacy_consistency.py \
            --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
            --genre "$genre" \
            --language "ar" \
            --pipeline-suffix ${pipeline_suffix} \
            --batch-size 32

    #     # --- Sense Metrics ---
        conda activate vllm
        read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "ar" --pipeline-suffix ${pipeline_suffix})

        # Build output path
        output_path_llm="/gscratch/stf/hardiksr/hiatus/metrics/ar/${genre}_${pipeline_suffix}_llm.jsonl"
        
        python scripts/5-evaluation/score_sense.py \
            --text-original-jsonl ${input_queries_path} \
            --text-privatized-jsonl ${privatized_queries_path} \
            --output-path ${output_path_llm} \
            --model-path "google/gemma-3-27b-it" 
    done
done


