#!/bin/bash

#SBATCH --job-name=obfuscate_eval_styleremix_authbench_zh
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=50:00:00 

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_styleremix_authbench_zh.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_styleremix_authbench_zh.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0

LANGUAGE="zh"
MODEL="meta-llama/Meta-Llama-3-8B"

num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}
genres=(
    "ecommerce_reviews"
    "literature"
    "media_reviews"
    "news"
    "poetry"
    "social_media"
)
METRICS_DIR="/gscratch/stf/hardiksr/hiatus/metrics/styleremix"

for ((i=0; i<total_settings; i++)); do
    echo "Processing setting $i of ${total_settings}"

    num_styles=${num_styles_to_change[$i]}
    echo "Processing num_styles: $num_styles"

    # Pipeline suffix.
    pipeline_suffix="obfuscate_eval_styleremix_authbench_zh-baseline-${num_styles}"

    for genre in "${genres[@]}"; do
        # Setup paths
        sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${genre}/authbench_${LANGUAGE}_${genre}_sampled_ids.json"

        # Run StyleRemix Baseline
        python scripts/7-baselines/styleremix/obfuscate_styleremix.py \
            --dataset authbench \
            --language "zh" \
            --genre "${genre}" \
            --top-n-styles-to-change "${num_styles}" \
            --output-suffix "${pipeline_suffix}" \
            --model-ckpt "${MODEL}" \
            --sampled-data-filepath "${sampled_data_filepath}"
    done
done


for num_styles in "${num_styles_to_change[@]}"; do
    PIPELINE_SUFFIX="obfuscate_eval_styleremix_authbench_zh-baseline-${num_styles}"
    for GENRE in "${genres[@]}"; do
        echo "Evaluating genre: ${GENRE}"

        sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${GENRE}/authbench_${LANGUAGE}_${GENRE}_sampled_ids.json"

        # AuthBench Eval
        echo "Running AuthBench Delta EER Eval for (${LANGUAGE}, ${GENRE})"
        conda activate vllm
        python scripts/7-baselines/authbench/authbench_eval.py \
            --model "${MODEL}" \
            --language "${LANGUAGE}" \
            --dataset authbench \
            --genre "${GENRE}" \
            --pipeline-suffix "${PIPELINE_SUFFIX}" \
            --output-dir "${METRICS_DIR}/authbench" \
            --batch-size 32 \
            --sampled-data-filepath "${sampled_data_filepath}"

        # Hiatus Eval
        echo "Running Hiatus Delta EER Eval for (${LANGUAGE}, ${GENRE})"
        conda activate hiatus-eval
        python scripts/5-evaluation/score_privacy_consistency.py \
            --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
            --genre "${GENRE}" \
            --language "${LANGUAGE}" \
            --pipeline-suffix "${PIPELINE_SUFFIX}" \
            --batch-size 32 \
            --sampled-data-filepath "${sampled_data_filepath}"
    done
done


echo "Accumulating Delta EER Metrics"
conda activate vllm

METHODS=()
for num_styles in "${num_styles_to_change[@]}"; do
    METHODS+=("obfuscate_eval_styleremix_authbench_zh-baseline-${num_styles}")
done

python scripts/5-evaluation/accumulate_delta_eer.py \
    --method "${METHODS[@]}" \
    --group-name "obfuscate_eval_styleremix_authbench_zh-baseline" \
    --dataset authbench \
    --language "${LANGUAGE}" \
    --model-tag "${MODEL//\//_}" \
    --genres "${genres[@]}" \
    --baseline-dir "${METRICS_DIR}" \
    --authbench-dir "${METRICS_DIR}/authbench" \
    --hiatus-dir "/gscratch/stf/hardiksr/hiatus/metrics/${LANGUAGE}" \
    --out-dir "/gscratch/stf/hardiksr/hiatus/metrics"
