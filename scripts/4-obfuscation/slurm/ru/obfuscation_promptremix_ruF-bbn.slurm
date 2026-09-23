#!/bin/bash
#SBATCH --job-name=obfuscation_promptremix-ruF-bbn
#SBATCH --mail-user=mpotto@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=10:00:00 

#SBATCH --chdir=/mmfs1/home/mpotto/code/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/mpotto/hiatus/logs/obfuscation_promptremix-ruF-bbn.out
#SBATCH --error=/gscratch/stf/mpotto/hiatus/logs/obfuscation_promptremix-ruF-bbn.err
set -e

source /mmfs1/home/mpotto/.bashrc
conda activate hiatus
module load cuda gcc

# Number of styles to update
num_styles_to_change=(3)
total_settings=${#num_styles_to_change[@]}

# Input queries path (SUBSTITUTE)
input_queries_path="/gscratch/ifml1/hiatus/hrs2/mode_perGenre-HRS2.101/TA3/hrs_06-27-24_russian_perGenre-HRS2.101/data/hrs_06-27-24_russian_perGenre-HRS2.101_TA3_input_queries.jsonl"

for ((i=0; i<total_settings; i++)); do
    echo "Processing setting $i of ${total_settings}"

    num_styles=${num_styles_to_change[$i]}
    echo "Processing num_styles: ${num_styles}"

    # Pipeline suffix 
    pipeline_suffix="promptremix-ruF-${num_styles}-bbn"

    # Privatized path (SUBSTITUTE)
    privatized_queries_path="/gscratch/stf/mpotto/hiatus/privatized/ru/hrs_06-27-24_russian_perGenre-HRS2.101_TA3_privatized_queries_${pipeline_suffix}.jsonl"
    
    # Temporary path for privatized text (SUBSTITUTE)
    tmp_path="/gscratch/scrubbed/mpotto/hiatus/privatized/ru/perGenre-HRS2.101_${num_styles}-bbn.json"

    # Insert PII
    python scripts/4-obfuscation/insert_pii.py \
        --input-path ${input_queries_path} \
        --input-key "fullText" \
        --lang "ru" \
        --name "Саша" \
        --output-key "PIIfullText" \
        --output-path ${tmp_path}

    # Run Prompt Remix
    python scripts/4-obfuscation/run_remix.py \
        --input-file ${tmp_path} \
        --model-id "Qwen/QwQ-32B" \
        --prompt-remix \
        --lang "ru" \
        --config-file "scripts/4-obfuscation/config/remix_config_ru.json" \
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
        --lang "ru" \
        --name "Саша" \

    # Rename columns
    python scripts/4-obfuscation/rename_columns.py \
        --input-path ${privatized_queries_path} \
        --output-path ${privatized_queries_path} \
        --old-key "PIIobfuscatedPostEditNoPII" \
        --new-key "fullText"
done
