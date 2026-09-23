#!/bin/bash

#SBATCH --job-name=obfuscation_promptremix-enF-style-transfer
#SBATCH --mail-user=kogolobo@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=24:00:00 

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-enF-style-transfer.out
#SBATCH --error=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-enF-style-transfer.err
set -e

source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda/12.8.1 gcc/12.3.0

# Number of styles to update
num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}

# Genres for English
en_genres=("perGenre-HRS2.1")

for ((i=0; i<total_settings; i++)); do
    conda activate dl
    echo "Processing setting $i of ${total_settings}"

    num_styles=${num_styles_to_change[$i]}
    echo "Processing num_styles: $num_styles"

    # Pipeline suffix.
    pipeline_suffix="promptremix-enF-style-transfer-${num_styles}"

    for genre in "${en_genres[@]}"; do
        # Setup paths
        read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre $genre --language "en" --pipeline-suffix ${pipeline_suffix})

        tmp_path="/gscratch/scrubbed/kogolobo/hiatus/privatized/en/${genre}_${num_styles}.json"
        echo "Input path: $input_queries_path"
        echo "Output path: $privatized_queries_path"
        echo "Tmp path: $tmp_path"
        

        # Insert PII
        python scripts/4-obfuscation/insert_pii.py \
            --input-path ${input_queries_path} \
            --input-key "fullText" \
            --lang "en" \
            --name "Alex" \
            --output-key "PIIfullText" \
            --output-path ${tmp_path}

        # Run Prompt Remix
        python scripts/4-obfuscation/run_remix.py \
            --style-transfer \
            --target-author-input-file "/gscratch/ifml1/kogolobo/style_transfer/en/${genre}/target_author_data.josnl" \
            --input-file ${tmp_path} \
            --model-id "Qwen/QwQ-32B" \
            --prompt-remix \
            --lang "en" \
            --config-file "scripts/4-obfuscation/config/remix_config_en.json" \
            --text-key "PIIfullText" \
            --author-key "authorIDs" \
            --document-key "documentID" \
            --top-n-styles-to-change "$num_styles" \
            --output-key "PIIobfuscated" \
            --output-file ${tmp_path} \

        # Remove PII
        python scripts/4-obfuscation/remove_pii.py \
            --input-path ${tmp_path} \
            --input-key "PIIobfuscated" \
            --output-key "PIIobfuscatedPostEditNoPII" \
            --output-path ${privatized_queries_path} \
            --lang "en" \
            --name "Alex" \

        # Rename columns
        python scripts/4-obfuscation/rename_columns.py \
            --input-path ${privatized_queries_path} \
            --output-path ${privatized_queries_path} \
            --old-key "PIIobfuscatedPostEditNoPII" \
            --new-key "fullText"

        conda activate hiatus-eval
        python scripts/5-evaluation/score_privacy_consistency.py \
            --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
            --genre "$genre" \
            --language "en" \
            --pipeline-suffix ${pipeline_suffix} \
            --batch-size 32
    done
done
