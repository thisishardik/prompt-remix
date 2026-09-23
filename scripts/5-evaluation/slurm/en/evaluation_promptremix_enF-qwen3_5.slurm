#!/bin/bash

#SBATCH --job-name=evaluation_promptremix_enF-qwen3.5
#SBATCH --mail-user=kogolobo@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=7
#SBATCH --time=50:00:00

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/kogolobo/hiatus/logs/evaluation_promptremix_enF-qwen3.5.out
#SBATCH --error=/gscratch/stf/kogolobo/hiatus/logs/evaluation_promptremix_enF-qwen3.5.err
set -e

source /mmfs1/home/kogolobo/.bashrc
module load cuda gcc

with_retry() {
    local max_attempts=3
    local delay=10 # Wait 10 seconds between attempts
    local attempt=1
    
    while (( attempt <= max_attempts )); do
        # Run the command passed to the function
        if "$@"; then
            return 0 # Success!
        else
            local exit_code=$?
            echo "--> [Warning] Command failed with exit code $exit_code (Attempt $attempt of $max_attempts)." >&2
            
            if (( attempt < max_attempts )); then
                echo "--> Retrying in $delay seconds..." >&2
                sleep $delay
            fi
            ((attempt++))
        fi
    done
    
    echo "--> [Error] Command failed permanently after $max_attempts attempts: $*" >&2
    return $exit_code # Pass the final failure code so `set -e` catches it
}

num_styles_to_change=(1 2 3)
en_genres=("perGenre-HRS2.1" "perGenre-HRS2.2" "perGenre-HRS2.3" "perGenre-HRS2.4" "perGenre-HRS2.5" "en_crossGenre" "perGenre-HRS3.1" "perGenre-HRS3.2" "perGenre-HRS3.3" "perGenre-HRS3.4" "perGenre-HRS3.5" "en_short_crossGenre")

for num_styles in "${num_styles_to_change[@]}"; do
    pipeline_suffix="promptremix-enF-qwen3.5-${num_styles}"

    for genre in "${en_genres[@]}"; do
        echo "Running evaluation for genre: ${genre}"

        # --- Privacy and Stylistic Consistency ---
        conda activate hiatus-eval
        with_retry python scripts/5-evaluation/score_privacy_consistency.py \
            --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
            --genre "$genre" \
            --language "en" \
            --pipeline-suffix ${pipeline_suffix} \
            --batch-size 32

        # --- Sense Metrics ---
        conda activate hiatus
        read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "en" --pipeline-suffix ${pipeline_suffix})

        output_path_llm="/gscratch/ifml1/kogolobo/metrics/en/${genre}_${pipeline_suffix}_llm.jsonl"

        with_retry python scripts/5-evaluation/score_sense.py \
            --text-original-jsonl ${input_queries_path} \
            --text-privatized-jsonl ${privatized_queries_path} \
            --output-path ${output_path_llm} \
            --model-path "mistralai/Mistral-Small-24B-Instruct-2501" \
            --batch-size 16
    done
done
