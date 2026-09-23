#!/bin/bash

#SBATCH --job-name=prompt_remix_obfuscation-ar4
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=110GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=60:00:00 
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscation_promptremix-ar4.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscation_promptremix-ar4.err

source /mmfs1/home/hardiksr/.bashrc
conda activate hiatus
module load cuda gcc coenv/jdk/17.0.8

# Number of styles to update
num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}

for ((i=0; i<total_settings; i++)); do
    echo "Processing setting $i of $total_settings"

    num_styles=${num_styles_to_change[$i]}
    echo "Processing num_styles: $num_styles"

    # Setup paths
    input_path="/gscratch/stf/hardiksr/hiatus/data/ar/evaluation/queries/hrs_06-27-24_arabic_crossGenre-combined_TA3_input_queries.jsonl"
    output_path="/gscratch/stf/hardiksr/hiatus/data/ar/obfuscation/hrs_06-27-24_arabic_crossGenre-combined_TA3_remix-ar4.jsonl"
    tmp_path="/gscratch/stf/hardiksr/hiatus/scrubbed/${num_styles}.json"

    echo "Running insert_pii script"
    # Run Insert Pii
    # python scripts/4-obfuscation/insert_pii.py \
    #     --input-path $input_path \
    #     --input-key "fullText" \
    #     --lang "ar" \
    #     --name "القائد" \
    #     --output-key "PIIfullText" \
    #     --output-path $tmp_path

    echo "Running run_remix script"
    # Run Prompt Remix
    python scripts/4-obfuscation/run_remix.py \
        --input-file $tmp_path \
        --model-id inceptionai/jais-adapted-70b-chat \
        --prompt-remix \
        --lang "ar" \
        --library "stanza" \
        --config-file "scripts/4-obfuscation/config/remix_config_ar.json" \
        --text-key "PIIfullText" \
        --author-key "authorIDs" \
        --document-key "documentID" \
        --top-n-styles-to-change "$num_styles" \
        --output-key "PIIobfuscated" \
        --output-file $tmp_path \
        --decoding-strategy "greedy" \
        # --temperature 0 \
        # --top-p 0 \
        # --top-k 0 \
        # --repetition-penalty 0

    echo "Running remove_pii script"
    # Remove PII
    python scripts/4-obfuscation/remove_pii.py \
        --input-path $tmp_path \
        --input-key "PIIobfuscated" \
        --output-key "PIIobfuscatedPostEditNoPII" \
        --output-path $output_path \
        --lang "ar" \
        --name "القائد" \

    echo "Running rename_columns script"
    # Rename columns
    python scripts/4-obfuscation/rename_columns.py \
        --input-path $output_path \
        --output-path $output_path \
        --old-key "PIIobfuscatedPostEditNoPII" \
        --new-key "fullText"
        
    done

