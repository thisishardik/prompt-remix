#!/bin/bash

#SBATCH --job-name=obfuscation_blog_en
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=24:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscation_blog_en.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscation_blog_en.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0 coenv/jdk/17.0.8

LANGUAGE="en"
SAMPLE_NAME="sample_2000"
NUM_STYLES_TO_CHANGE=(1 2 3)
PII_NAME="Alex"
MODEL_ID="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
CONFIG_FILE="scripts/4-obfuscation/config/remix_config_en.json"
LIBRARY="spacy"

for num_styles in "${NUM_STYLES_TO_CHANGE[@]}"; do
    echo "Processing Blog Authorship, num_styles: ${num_styles}"

    PIPELINE_SUFFIX="promptremix-blog-qwen3.6-${num_styles}"

    read -r input_queries_path privatized_queries_path candidate_queries_path ground_truth_path \
        < <(python scripts/7-baselines/blog_authorship/get_blog_paths.py \
            --sample-name ${SAMPLE_NAME} \
            --pipeline-suffix ${PIPELINE_SUFFIX})

    mkdir -p "$(dirname ${privatized_queries_path})"

    tmp_path="/gscratch/scrubbed/hardiksr/hiatus/privatized/blog_authorship/${SAMPLE_NAME}_${num_styles}_${PIPELINE_SUFFIX}.json"
    mkdir -p "$(dirname ${tmp_path})"

    echo "Input queries path: ${input_queries_path}"
    echo "Privatized queries path: ${privatized_queries_path}"
    echo "Temporary path: ${tmp_path}"

    # Insert PII
    python scripts/4-obfuscation/insert_pii.py \
        --input-path "${input_queries_path}" \
        --input-key "fullText" \
        --lang "${LANGUAGE}" \
        --name "${PII_NAME}" \
        --output-key "PIIfullText" \
        --output-path "${tmp_path}"

    # Run Prompt Remix
    python scripts/4-obfuscation/run_remix.py \
        --input-file "${tmp_path}" \
        --model-id "${MODEL_ID}" \
        --prompt-remix \
        --lang "${LANGUAGE}" \
        --library "${LIBRARY}" \
        --config-file "${CONFIG_FILE}" \
        --text-key "PIIfullText" \
        --author-key "authorIDs" \
        --document-key "documentID" \
        --top-n-styles-to-change "${num_styles}" \
        --output-key "PIIobfuscated" \
        --max-gpu-memory-utilization 0.475 \
        --output-file "${tmp_path}"

    # Remove PII
    python scripts/4-obfuscation/remove_pii.py \
        --input-path "${tmp_path}" \
        --input-key "PIIobfuscated" \
        --output-key "PIIobfuscatedPostEditNoPII" \
        --output-path "${privatized_queries_path}" \
        --lang "${LANGUAGE}" \
        --name "${PII_NAME}"

    # Rename columns
    python scripts/4-obfuscation/rename_columns.py \
        --input-path "${privatized_queries_path}" \
        --output-path "${privatized_queries_path}" \
        --old-key "PIIobfuscatedPostEditNoPII" \
        --new-key "fullText"

done
