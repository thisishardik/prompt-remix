#!/bin/bash

#SBATCH --job-name=evaluation_promptremix_ruF
#SBATCH --mail-user=mpotto@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=8:00:00

#SBATCH --chdir=/mmfs1/home/mpotto/code/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/mpotto/hiatus/logs/evaluation_promptremix_ruF.out
#SBATCH --error=/gscratch/stf/mpotto/hiatus/logs/evaluation_promptremix_ruF.err
set -e

source /mmfs1/home/mpotto/.bashrc
module load cuda gcc

num_styles_to_change=(1 2 3)
ru_genres=("perGenre-HRS2.101" "perGenre-HRS2.102" "perGenre-HRS2.103" "perGenre-HRS2.104" "perGenre-HRS2.105" "ru_crossGenre")

for num_styles in "${num_styles_to_change[@]}"; do
    pipeline_suffix="promptremix-ruF-${num_styles}"

    for genre in "${ru_genres[@]}"; do
        echo "Running evaluation for genre: ${genre}"

        # --- Privacy and Stylistic Consistency ---
        conda activate hiatus-eval
        python scripts/5-evaluation/score_privacy_consistency.py \
            --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
            --genre "$genre" \
            --language "ru" \
            --pipeline-suffix ${pipeline_suffix} \
            --batch-size 32

        # --- Sense Metrics ---
        conda activate hiatus
        read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "ru" --pipeline-suffix ${pipeline_suffix})

        # Build output path
        output_path_llm="/gscratch/ifml1/hiatus/metrics/ru/${genre}_${pipeline_suffix}_llm.jsonl"
        
        python scripts/5-evaluation/score_sense.py \
            --text-original-jsonl ${input_queries_path} \
            --text-privatized-jsonl ${privatized_queries_path} \
            --output-path ${output_path_llm} \
            --model-path "mistralai/Mistral-Small-24B-Instruct-2501" 
    done
done


