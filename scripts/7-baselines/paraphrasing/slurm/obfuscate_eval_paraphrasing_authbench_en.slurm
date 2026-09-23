#!/bin/bash

#SBATCH --job-name=obfuscate_eval_paraphrasing_authbench_en
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=70GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=5
#SBATCH --time=120:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_paraphrasing_authbench_en.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_paraphrasing_authbench_en.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda gcc

LANGUAGE="en"
MODEL="Qwen/Qwen2.5-3B-Instruct"
PIPELINE_SUFFIX="paraphrasing"
METRICS_DIR="/gscratch/stf/hardiksr/hiatus/metrics/paraphrasing"

genres=(
    "blog"
    "ecommerce_reviews"
    "literature"
    "news"
    "poetry"
    "qna"
    "research_paper"
)

for GENRE in "${genres[@]}"; do
    echo "Obfuscating genre: ${GENRE}"

    sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${GENRE}/authbench_${LANGUAGE}_${GENRE}_sampled_ids.json"

    conda activate vllm
    python scripts/7-baselines/paraphrasing/obfuscate_paraphrasing.py \
        --dataset authbench \
        --language "${LANGUAGE}" \
        --genre "${GENRE}" \
        --output-suffix "${PIPELINE_SUFFIX}" \
        --sampled-data-filepath "${sampled_data_filepath}"
done

for GENRE in "${genres[@]}"; do
    echo "Evaluating genre: ${GENRE}"
    
    sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${GENRE}/authbench_${LANGUAGE}_${GENRE}_sampled_ids.json"

    # AuthBench Eval
    echo "Running AuthBench Delta EER Eval for Paraphrasing (${LANGUAGE}, ${GENRE})"
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
    echo "Running Hiatus Delta EER Eval for Paraphrasing (${LANGUAGE}, ${GENRE})"
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
    --group-name "${PIPELINE_SUFFIX}_authbench_en-baseline" \
    --dataset authbench \
    --language "${LANGUAGE}" \
    --model-tag "${MODEL//\//_}" \
    --genres "${genres[@]}" \
    --baseline-dir "${METRICS_DIR}" \
    --authbench-dir "${METRICS_DIR}/authbench" \
    --hiatus-dir "/gscratch/stf/hardiksr/hiatus/metrics/${LANGUAGE}" \
    --out-dir "/gscratch/stf/hardiksr/hiatus/metrics"
