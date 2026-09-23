#!/bin/bash

#SBATCH --job-name=obfuscation_promptremix-tbt_ar0
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=120GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=40:00:00 

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscation_promptremix-tbt_ar0.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscation_promptremix-tbt_ar0.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate hiatus
module load cuda gcc coenv/jdk/17.0.8


# Number of styles to update
num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}

# Genres for Arabic
ar_genres=("AMINA" "bbn_aljazeera_blog_1")

for ((i=0; i<total_settings; i++)); do
    echo "Processing setting $i of ${total_settings}"

    num_styles=${num_styles_to_change[$i]}
    echo "Processing num_styles: ${num_styles}"

    # Setup paths
    pipeline_suffix="promptremix-tbt_ar0-${num_styles}"

    for genre in "${ar_genres[@]}"; do
         # Setup paths
        read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "ar" --pipeline-suffix ${pipeline_suffix})
        
        translated_queries_path=$(python scripts/4-obfuscation/get_translated_paths.py \
            --input-path ${input_queries_path} \
            --source "ar" \
            --target "en")
        
        tmp_path="/gscratch/scrubbed/hardiksr/hiatus/privatized/ar/${genre}_${num_styles}.json"

        # Translate
        python scripts/4-obfuscation/translate.py \
            --source "ar" \
            --target "en" \
            --field "fullText" \
            --input-path ${input_queries_path} \
            --output-path ${translated_queries_path} \
            --backend "google"

        # Insert PII
        python scripts/4-obfuscation/insert_pii.py \
            --input-path ${translated_queries_path} \
            --input-key "fullText" \
            --lang "en" \
            --name "Mika" \
            --output-key "PIIfullText" \
            --output-path ${tmp_path}

        # Run Prompt Remix
        python scripts/4-obfuscation/run_remix.py \
            --input-file ${tmp_path} \
            --model-id "Qwen/QwQ-32B" \
            --prompt-remix \
            --lang "en" \
            --config-file "/mmfs1/home/hardiksr/repository/hiatus/scripts/4-obfuscation/config/remix_config_en.json" \
            --text-key "PIIfullText" \
            --author-key "authorIDs" \
            --document-key "documentID" \
            --top-n-styles-to-change "$num_styles" \
            --output-key "PIIobfuscated" \
            --output-file ${tmp_path} \
            --decoding-strategy "default"

        # Remove PII
        python scripts/4-obfuscation/remove_pii.py \
            --input-path ${tmp_path} \
            --input-key "PIIobfuscated" \
            --output-key "PIIobfuscatedPostEditNoPII" \
            --output-path ${privatized_queries_path} \
            --lang "en" \
            --name "Mika"

        # Translate
        python scripts/4-obfuscation/translate.py \
            --source "en" \
            --target "ar" \
            --field "PIIobfuscatedPostEditNoPII" \
            --backtranslate \
            --input-path ${privatized_queries_path} \
            --backend "google"

        # Rename columns
        python scripts/4-obfuscation/rename_columns.py \
            --input-path ${privatized_queries_path} \
            --output-path ${privatized_queries_path} \
            --old-key "PIIobfuscatedPostEditNoPII" \
            --new-key "fullText"
        done
done
