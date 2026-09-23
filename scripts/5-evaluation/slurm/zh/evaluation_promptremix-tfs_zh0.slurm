#!/bin/bash

#SBATCH --job-name=evaluation_promptremix-zh0
#SBATCH --mail-user=mpotto@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=80:00:00

#SBATCH --chdir=/mmfs1/home/mpotto/code/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/mpotto/hiatus/logs/evaluation_promptremix-zh0.out
#SBATCH --error=/gscratch/stf/mpotto/hiatus/logs/evaluation_promptremix-zh0.err
set -e

source /mmfs1/home/mpotto/.bashrc
module load cuda gcc

num_styles_to_change=(1 2 3)
zh_genres=("perGenre-HRS3.301" "perGenre-HRS3.302" "perGenre-HRS3.303" "perGenre-HRS3.304" "perGenre-HRS3.305" "zh_short_crossGenre" "zh_medium_crossGenre")

for num_styles in "${num_styles_to_change[@]}"; do
    pipeline_suffix="promptremix-zh0-${num_styles}"

    for genre in "${zh_genres[@]}"; do
        echo "Running evaluation for genre: ${genre}"

        # --- Privacy and Stylistic Consistency ---
        conda activate hiatus-eval
        python scripts/5-evaluation/score_privacy_consistency.py \
            --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
            --genre "$genre" \
            --language "zh" \
            --pipeline-suffix ${pipeline_suffix} \
            --batch-size 32

        # --- Sense Metrics ---
        conda activate hiatus
        read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "zh" --pipeline-suffix ${pipeline_suffix})

        # Build output path
        output_path_llm="/gscratch/ifml1/hiatus/metrics/zh/${genre}_${pipeline_suffix}_llm.jsonl"
        
        python scripts/5-evaluation/score_sense.py \
            --text-original-jsonl ${input_queries_path} \
            --text-privatized-jsonl ${privatized_queries_path} \
            --output-path ${output_path_llm} \
            --model-path "mistralai/Mistral-Small-24B-Instruct-2501" 
    done
done


