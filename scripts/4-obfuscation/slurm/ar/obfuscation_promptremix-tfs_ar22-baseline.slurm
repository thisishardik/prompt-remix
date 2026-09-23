#!/bin/bash

#SBATCH --job-name=obfuscation_promptremix-tfs_ar22-baseline
#SBATCH --mail-user=kogolobo@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=50:00:00

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-tfs_ar22-baseline.out
#SBATCH --error=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-tfs_ar22-baseline.err
set -e

source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda/12.8.1 gcc/12.3.0

# Number of styles to update
num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}

# Genres for Arabic
ar_genres=("ar_twitter_1" "saudinewsnet" "AMINA" "bbn_aljazeera_blog_1")

for ((i=0; i<total_settings; i++)); do
    echo "Processing setting $i of ${total_settings}"

    num_styles=${num_styles_to_change[$i]}
    echo "Processing num_styles: ${num_styles}"

    pipeline_suffix="promptremix-tfs_ar22-baseline-${num_styles}"

    for genre in "${ar_genres[@]}"; do
        read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "ar" --pipeline-suffix ${pipeline_suffix})

        tmp_path="/gscratch/scrubbed/kogolobo/hiatus/privatized/ar/${genre}_${num_styles}_${pipeline_suffix}.json"
        echo "Input queries path: ${input_queries_path}"
        echo "Privatized queries path: ${privatized_queries_path}"
        echo "Temporary path: ${tmp_path}"

        # Insert PII
        python scripts/4-obfuscation/insert_pii.py \
            --input-path ${input_queries_path} \
            --input-key "fullText" \
            --lang "ar" \
            --name "نور" \
            --output-key "PIIfullText" \
            --output-path ${tmp_path}

        # Run Prompt Remix (no guided decoding, no random directions)
        python scripts/4-obfuscation/run_remix.py \
            --input-file ${tmp_path} \
            --model-id "Qwen/Qwen2.5-72B-Instruct" \
            --prompt-remix \
            --lang "ar" \
            --library "corenlp" \
            --config-file "scripts/4-obfuscation/config/remix_config_ar3.json" \
            --text-key "PIIfullText" \
            --author-key "authorIDs" \
            --document-key "documentID" \
            --top-n-styles-to-change "${num_styles}" \
            --output-key "PIIobfuscated" \
            --output-file ${tmp_path}

        # Remove PII
        python scripts/4-obfuscation/remove_pii.py \
            --input-path ${tmp_path} \
            --input-key "PIIobfuscated" \
            --output-key "PIIobfuscatedPostEditNoPII" \
            --output-path ${privatized_queries_path} \
            --lang "ar" \
            --name "نور"

        # Rename columns
        python scripts/4-obfuscation/rename_columns.py \
            --input-path ${privatized_queries_path} \
            --output-path ${privatized_queries_path} \
            --old-key "PIIobfuscatedPostEditNoPII" \
            --new-key "fullText"
    done
done
