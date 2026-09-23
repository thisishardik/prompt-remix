#!/bin/bash

#SBATCH --job-name=obfuscate_eval_round_trip_mt_authbench_zh
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=85GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=120:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_round_trip_mt_authbench_zh.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_round_trip_mt_authbench_zh.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda gcc

LANGUAGE="zh"
MODEL="Qwen/Qwen2.5-3B-Instruct"
PIPELINE_SUFFIX="round_trip_mt"
METRICS_DIR="/gscratch/stf/hardiksr/hiatus/metrics/round_trip_mt"

genres=(
    "ecommerce_reviews"
    "literature"
    "media_reviews"
    "news"
    "poetry"
    "social_media"
)

for GENRE in "${genres[@]}"; do
    echo "Obfuscating genre: ${GENRE}"

    sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${GENRE}/authbench_${LANGUAGE}_${GENRE}_sampled_ids.json"
    conda activate vllm
    python scripts/7-baselines/round_trip_mt/obfuscate_round_trip_mt.py \
        --dataset authbench \
        --language "${LANGUAGE}" \
        --genre "${GENRE}" \
        --output-suffix "${PIPELINE_SUFFIX}" \
        --sampled-data-filepath "${sampled_data_filepath}"
done

for GENRE in "${genres[@]}"; do
    echo "Evaluating genre: ${GENRE}"

    # AuthBench Eval
    echo "Running AuthBench Delta EER Eval for Round-Trip MT (${LANGUAGE}, ${GENRE})"
    sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${GENRE}/authbench_${LANGUAGE}_${GENRE}_sampled_ids.json"
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
    echo "Running Hiatus Delta EER Eval for Round-Trip MT (${LANGUAGE}, ${GENRE})"
    sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${GENRE}/authbench_${LANGUAGE}_${GENRE}_sampled_ids.json"
    conda activate hiatus-eval
    python scripts/5-evaluation/score_privacy_consistency.py \
        --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
        --genre "${GENRE}" \
        --language "${LANGUAGE}" \
        --pipeline-suffix "${PIPELINE_SUFFIX}" \
        --batch-size 32 \
        --dataset authbench \
        --sampled-data-filepath "${sampled_data_filepath}"
done

echo "Accumulating Delta EER Metrics"

conda activate vllm
python scripts/5-evaluation/accumulate_delta_eer.py \
    --method "${PIPELINE_SUFFIX}" \
    --group-name "${PIPELINE_SUFFIX}_authbench_zh-baseline" \
    --dataset authbench \
    --language "${LANGUAGE}" \
    --model-tag "${MODEL//\//_}" \
    --genres "${genres[@]}" \
    --baseline-dir "${METRICS_DIR}" \
    --authbench-dir "${METRICS_DIR}/authbench" \
    --hiatus-dir "/gscratch/stf/hardiksr/hiatus/metrics/${LANGUAGE}" \
    --out-dir "/gscratch/stf/hardiksr/hiatus/metrics"
