#!/bin/bash

#SBATCH --job-name=obfuscation_authbench_zh
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=24:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscation_authbench_zh.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscation_authbench_zh.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0 coenv/jdk/17.0.8

LANGUAGE="zh"
NUM_STYLES_TO_CHANGE=(1 2 3)
PII_NAME="小明"
MODEL_ID="RedHatAI/Qwen3.6-35B-A3B-NVFP4"
CONFIG_FILE="scripts/4-obfuscation/config/remix_config_zh.json"
LIBRARY="stanza"

ZH_GENRES=(
    "social_media" "poetry" "ecommerce_reviews" "literature" "media_reviews" "news"
)

for num_styles in "${NUM_STYLES_TO_CHANGE[@]}"; do
    for genre in "${ZH_GENRES[@]}"; do
        echo "========================================"
        echo "Processing genre: ${genre}, num_styles: ${num_styles}"
        echo "========================================"

        PIPELINE_SUFFIX="promptremix-authbench-qwen3.6-${num_styles}"
        GENRE_SAFE=$(echo "${genre}" | tr '/' '_')

        read -r input_queries_path privatized_queries_path candidate_queries_path ground_truth_path \
            < <(python scripts/7-baselines/authbench/get_authbench_paths.py \
                --language ${LANGUAGE} \
                --genre "${genre}" \
                --pipeline-suffix ${PIPELINE_SUFFIX})

        mkdir -p "$(dirname ${privatized_queries_path})"

        tmp_path="/gscratch/scrubbed/hardiksr/hiatus/privatized/authbench/${LANGUAGE}/${GENRE_SAFE}_${num_styles}_${PIPELINE_SUFFIX}.json"
        mkdir -p "$(dirname ${tmp_path})"

        echo "Input queries path: ${input_queries_path}"
        echo "Privatized queries path: ${privatized_queries_path}"
        echo "Temporary path: ${tmp_path}"

        python scripts/4-obfuscation/insert_pii.py \
            --input-path "${input_queries_path}" \
            --input-key "fullText" \
            --lang "${LANGUAGE}" \
            --name "${PII_NAME}" \
            --output-key "PIIfullText" \
            --output-path "${tmp_path}"

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

        python scripts/4-obfuscation/remove_pii.py \
            --input-path "${tmp_path}" \
            --input-key "PIIobfuscated" \
            --output-key "PIIobfuscatedPostEditNoPII" \
            --output-path "${privatized_queries_path}" \
            --lang "${LANGUAGE}" \
            --name "${PII_NAME}"

        python scripts/4-obfuscation/rename_columns.py \
            --input-path "${privatized_queries_path}" \
            --output-path "${privatized_queries_path}" \
            --old-key "PIIobfuscatedPostEditNoPII" \
            --new-key "fullText"

        echo "Done with ${genre}, num_styles=${num_styles}"
    done
done

