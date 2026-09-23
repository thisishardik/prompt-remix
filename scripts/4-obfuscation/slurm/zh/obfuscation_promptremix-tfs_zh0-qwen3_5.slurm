#!/bin/bash

#SBATCH --job-name=obfuscation_promptremix-tfs_zh0-qwen3.5
#SBATCH --mail-user=kogolobo@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=24:00:00 

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-tfs_zh0-qwen3.5.out
#SBATCH --error=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-tfs_zh0-qwen3.5.err
set -e

source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda/12.8.1 gcc/12.3.0

# Number of styles to update
num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}

# Genres for Chinese
zh_genres=("perGenre-HRS3.301" "perGenre-HRS3.302" "perGenre-HRS3.303" "perGenre-HRS3.304" "perGenre-HRS3.305" "zh_short_crossGenre" "zh_medium_crossGenre")

for ((i=0; i<total_settings; i++)); do
    echo "Processing setting $i of ${total_settings}"

    num_styles=${num_styles_to_change[$i]}
    echo "Processing num_styles: ${num_styles}"

    # Pipeline suffix.
    pipeline_suffix="promptremix-zh0-qwen3.5-${num_styles}"

    # Iterate over genres.
    for genre in "${zh_genres[@]}"; do
        # Setup paths
        read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "zh" --pipeline-suffix ${pipeline_suffix})

        tmp_path="/gscratch/scrubbed/kogolobo/hiatus/privatized/zh/${genre}_${num_styles}_${pipeline_suffix}.json"
        echo "Input queries path: ${input_queries_path}"
        echo "Privatized queries path: ${privatized_queries_path}"
        echo "Temporary path: ${tmp_path}"

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
            --name "李晓明" \

        # Rename columns
        python scripts/4-obfuscation/rename_columns.py \
            --input-path ${privatized_queries_path} \
            --output-path ${privatized_queries_path} \
            --old-key "PIIobfuscatedPostEditNoPII" \
            --new-key "fullText"
    done
done
