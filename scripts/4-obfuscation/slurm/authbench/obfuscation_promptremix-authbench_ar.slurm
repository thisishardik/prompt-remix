#!/bin/bash

#SBATCH --job-name=promptremix_obfuscation-authbench_ar
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=90GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=7
#SBATCH --time=100:00:00
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscation_promptremix-authbench_ar.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscation_promptremix-authbench_ar.err

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda gcc coenv/jdk/17.0.8

# Number of styles to update
num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}
genres=("social_media" "poetry" "literature" "news")

for ((i=0; i<total_settings; i++)); do
    echo "Processing setting $i of $total_settings"

    num_styles=${num_styles_to_change[$i]}
    echo "Processing num_styles: $num_styles"

    pipeline_suffix="promptremix_authbench-ar0-${num_styles}"

    for genre in "${genres[@]}"; do
        # Setup paths
        read -r input_queries_path privatized_queries_path candidate_queries_path ground_truth_path < <(python scripts/7-baselines/authbench/get_authbench_paths.py --language "ar" --genre ${genre} --pipeline-suffix ${pipeline_suffix})

        tmp_path="/gscratch/stf/hardiksr/hiatus/scrubbed/${genre}_${num_styles}_${pipeline_suffix}.json"
        sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/ar/${genre}/authbench_ar_${genre}_sampled_ids.json"
        
        echo "Input queries path: ${input_queries_path}"
        echo "Privatized queries path: ${privatized_queries_path}"
        echo "Temporary path: ${tmp_path}"
        echo "Sampled data path: ${sampled_data_filepath}"

        echo "Running insert_pii script"
        # Run Insert Pii
        python scripts/4-obfuscation/insert_pii.py \
            --input-path ${input_queries_path} \
            --input-key "fullText" \
            --lang "ar" \
            --name "نور" \
            --output-key "PIIfullText" \
            --output-path ${tmp_path} \
            --sampled-data-filepath "${sampled_data_filepath}"

        # Run Prompt Remix
        python scripts/4-obfuscation/run_remix.py \
            --input-file ${tmp_path} \
            --model-id "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" \
            --prompt-remix \
            --lang "ar" \
            --library "corenlp" \
            --config-file "scripts/4-obfuscation/config/remix_config_ar3.json" \
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

