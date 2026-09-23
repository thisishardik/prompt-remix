#!/bin/bash

#SBATCH --job-name=obfuscation_promptremix-tfs_zh0-transfer-qwen3_5
#SBATCH --mail-user=kogolobo@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=24:00:00 

# Array directive: spawns 3 jobs with indices 0, 1, and 2
#SBATCH --array=0-2

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/hiatus
#SBATCH --export=all

# Modified to include %A (job ID) and %a (array task ID) to prevent log collision
#SBATCH --output=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-tfs_zh0-transfer-qwen3_5_%A_%a.out
#SBATCH --error=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-tfs_zh0-transfer-qwen3_5_%A_%a.err

set -e

source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda/12.8.1 gcc/12.3.0

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

# Number of styles to update
num_styles_to_change=(1 2 3)

# Use the SLURM_ARRAY_TASK_ID to pick the specific setting for this job instance
num_styles=${num_styles_to_change[$SLURM_ARRAY_TASK_ID]}
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID} | Processing num_styles: ${num_styles}"

# Genres for Chinese
zh_genres=("perGenre-HRS3.301" "perGenre-HRS3.302" "perGenre-HRS3.303" "perGenre-HRS3.304" "perGenre-HRS3.305" "zh_short_crossGenre" "zh_medium_crossGenre")

# Pipeline suffix.
pipeline_suffix="promptremix-zh0-transfer-qwen3_5-${num_styles}"

# Iterate over genres.
for genre in "${zh_genres[@]}"; do
    # Setup paths
    read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "zh" --pipeline-suffix ${pipeline_suffix})

    tmp_path="/gscratch/scrubbed/kogolobo/hiatus/privatized/zh/${genre}_${num_styles}_${pipeline_suffix}.json"
    echo "Input queries path: ${input_queries_path}"
    echo "Privatized queries path: ${privatized_queries_path}"
    echo "Temporary path: ${tmp_path}"

    conda activate dl
    # Insert PII
    python scripts/4-obfuscation/insert_pii.py \
        --input-path ${input_queries_path} \
        --input-key "fullText" \
        --lang "zh" \
        --name "李晓明" \
        --output-key "PIIfullText" \
        --output-path ${tmp_path}

    # Run Prompt Remix
    python scripts/4-obfuscation/run_remix.py \
        --style-transfer \
        --target-author-input-file "/gscratch/ifml1/kogolobo/style_transfer/zh/${genre}/target_author_data.json" \
        --input-file ${tmp_path} \
        --model-id "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" \
        --prompt-remix \
        --lang "zh" \
        --config-file "scripts/4-obfuscation/config/remix_config_zh.json" \
        --text-key "PIIfullText" \
        --author-key "authorIDs" \
        --document-key "documentID" \
        --top-n-styles-to-change "${num_styles}" \
        --output-key "PIIobfuscated" \
        --max-gpu-memory-utilization 0.475 \
        --output-file ${tmp_path}

    # Remove PII
    python scripts/4-obfuscation/remove_pii.py \
        --input-path ${tmp_path} \
        --input-key "PIIobfuscated" \
        --output-key "PIIobfuscatedPostEditNoPII" \
        --output-path ${privatized_queries_path} \
        --lang "zh" \
        --name "李晓明"

    # Rename columns
    python scripts/4-obfuscation/rename_columns.py \
        --input-path ${privatized_queries_path} \
        --output-path ${privatized_queries_path} \
        --old-key "PIIobfuscatedPostEditNoPII" \
        --new-key "fullText"

    # Evaluate
    echo "Running evaluation for genre: ${genre}"

    # --- Privacy and Stylistic Consistency ---
    conda activate hiatus-eval
    with_retry python scripts/5-evaluation/score_privacy_consistency.py \
        --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
        --genre "$genre" \
        --language "zh" \
        --pipeline-suffix ${pipeline_suffix} \
        --batch-size 32

    # --- Sense Metrics ---
    conda activate hiatus
    read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "zh" --pipeline-suffix ${pipeline_suffix})

    output_path_llm="/gscratch/ifml1/kogolobo/metrics/zh/${genre}_${pipeline_suffix}_llm.jsonl"

    with_retry python scripts/5-evaluation/score_sense.py \
        --text-original-jsonl ${input_queries_path} \
        --text-privatized-jsonl ${privatized_queries_path} \
        --output-path ${output_path_llm} \
        --model-path "mistralai/Mistral-Small-24B-Instruct-2501" \
        --batch-size 16
done