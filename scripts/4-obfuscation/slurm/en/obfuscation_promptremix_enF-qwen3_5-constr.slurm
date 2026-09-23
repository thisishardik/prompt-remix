#!/bin/bash

#SBATCH --job-name=obfuscation_promptremix-enF-qwen3.5-constr
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
#SBATCH --output=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-enF-qwen3.5-constr.out
#SBATCH --error=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-enF-qwen3.5-constr.err
set -e

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

source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda/12.8.1 gcc/12.3.0

# Number of styles to update
num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}

# Genres for English
en_genres=("perGenre-HRS2.1" "perGenre-HRS2.2" "perGenre-HRS2.3" "perGenre-HRS2.4" "perGenre-HRS2.5" "en_crossGenre" "perGenre-HRS3.1" "perGenre-HRS3.2" "perGenre-HRS3.3" "perGenre-HRS3.4" "perGenre-HRS3.5" "en_short_crossGenre")

for ((i=0; i<total_settings; i++)); do
    echo "Processing setting $i of ${total_settings}"

    num_styles=${num_styles_to_change[$i]}
    echo "Processing num_styles: ${num_styles}"

    pipeline_suffix="promptremix-enF-qwen3.5-constr-${num_styles}"

    for genre in "${en_genres[@]}"; do
        read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "en" --pipeline-suffix ${pipeline_suffix})

        tmp_path="/gscratch/scrubbed/kogolobo/hiatus/privatized/en/${genre}_${num_styles}_${pipeline_suffix}.json"
        echo "Input queries path: ${input_queries_path}"
        echo "Privatized queries path: ${privatized_queries_path}"
        echo "Temporary path: ${tmp_path}"

        conda activate dl
        # Insert PII
        python scripts/4-obfuscation/insert_pii.py \
            --input-path ${input_queries_path} \
            --input-key "fullText" \
            --lang "en" \
            --name "Alex" \
            --output-key "PIIfullText" \
            --output-path ${tmp_path}

        # Run Prompt Remix with constrained decoding
        python scripts/4-obfuscation/run_remix.py \
            --input-file ${tmp_path} \
            --model-id "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" \
            --prompt-remix \
            --lang "en" \
            --config-file "scripts/4-obfuscation/config/remix_config_en.json" \
            --text-key "PIIfullText" \
            --author-key "authorIDs" \
            --document-key "documentID" \
            --top-n-styles-to-change "${num_styles}" \
            --output-key "PIIobfuscated" \
            --max-gpu-memory-utilization 0.475 \
            --use-constrained-decoding \
            --output-file ${tmp_path}

        # Remove PII
        python scripts/4-obfuscation/remove_pii.py \
            --input-path ${tmp_path} \
            --input-key "PIIobfuscated" \
            --output-key "PIIobfuscatedPostEditNoPII" \
            --output-path ${privatized_queries_path} \
            --lang "en" \
            --name "Alex"

        # Rename columns
        python scripts/4-obfuscation/rename_columns.py \
            --input-path ${privatized_queries_path} \
            --output-path ${privatized_queries_path} \
            --old-key "PIIobfuscatedPostEditNoPII" \
            --new-key "fullText"

        # Evaluate
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