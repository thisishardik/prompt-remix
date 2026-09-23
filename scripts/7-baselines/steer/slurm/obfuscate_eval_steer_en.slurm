#!/bin/bash

#SBATCH --job-name=obfuscate_eval_steer_en
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=120:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_steer_en.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_steer_en.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda gcc

LANGUAGE="en"
MODEL="Qwen/Qwen2.5-3B-Instruct"
PIPELINE_SUFFIX="steer"
METRICS_DIR="/gscratch/stf/hardiksr/hiatus/metrics/steer"
TARGET_STYLE="english_tweet"

en_genres=(
    "perGenre-HRS2.3"
    "perGenre-HRS2.4"
    "perGenre-HRS2.5"
    "en_crossGenre"
    "perGenre-HRS3.1"
    "perGenre-HRS3.2"
    "perGenre-HRS3.3"
    "perGenre-HRS3.4"
    "perGenre-HRS3.5"
    "en_short_crossGenre"
)

for GENRE in "${en_genres[@]}"; do
    echo "Obfuscating genre: ${GENRE}"

    conda activate vllm
    python scripts/7-baselines/steer/obfuscate_steer.py \
        --dataset hrs \
        --language "${LANGUAGE}" \
        --genre "${GENRE}" \
        --output-suffix "${PIPELINE_SUFFIX}" \
        --target-style "${TARGET_STYLE}"
done

for GENRE in "${en_genres[@]}"; do
    echo "Evaluating genre: ${GENRE}"

    # AuthBench Eval
    echo "Running AuthBench Delta EER Eval for STEER (${LANGUAGE}, ${GENRE})"
    conda activate vllm
    python scripts/7-baselines/authbench/authbench_eval.py \
        --model "${MODEL}" \
        --language "${LANGUAGE}" \
        --dataset hrs \
        --genre "${GENRE}" \
        --pipeline-suffix "${PIPELINE_SUFFIX}" \
        --output-dir "${METRICS_DIR}/authbench" \
        --batch-size 32

    # Hiatus Eval
    echo "Running Hiatus Delta EER Eval for STEER (${LANGUAGE}, ${GENRE})"
    conda activate hiatus-eval
    python scripts/5-evaluation/score_privacy_consistency.py \
        --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
        --genre "${GENRE}" \
        --language "${LANGUAGE}" \
        --pipeline-suffix "${PIPELINE_SUFFIX}" \
        --batch-size 32
done

echo "Accumulating Delta EER Metrics"

conda activate vllm
python scripts/5-evaluation/accumulate_delta_eer.py \
    --method "${PIPELINE_SUFFIX}" \
    --language "${LANGUAGE}" \
    --model-tag "Qwen_Qwen2.5-3B-Instruct" \
    --genres "${en_genres[@]}" \
    --baseline-dir "${METRICS_DIR}" \
    --authbench-dir "${METRICS_DIR}/authbench" \
    --hiatus-dir "/gscratch/stf/hardiksr/hiatus/metrics/${LANGUAGE}" \
    --out-dir "/gscratch/stf/hardiksr/hiatus/metrics"
