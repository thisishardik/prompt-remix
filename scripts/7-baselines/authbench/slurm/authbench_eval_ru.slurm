#!/bin/bash

#SBATCH --job-name=authbench_eval_ru
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=24:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/authbench_eval_ru.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/authbench_eval_ru.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda gcc

LANGUAGE="ru"
OUTPUT_DIR="/gscratch/stf/hardiksr/hiatus/metrics/authbench"
MODELS=("meta-llama/Meta-Llama-3-8B" "Qwen/Qwen2.5-3B-Instruct")
NUM_STYLES_TO_CHANGE=(1 2 3)

GENRES=(
    "social_media" "qna" "poetry" "literature" "news"
)

for model in "${MODELS[@]}"; do
    for genre in "${GENRES[@]}"; do
        echo "Model: ${model}, Genre: ${genre}"

        python scripts/7-baselines/authbench/authbench_eval.py \
            --model "${model}" \
            --language "${LANGUAGE}" \
            --genre "${genre}" \
            --output-dir "${OUTPUT_DIR}/${LANGUAGE}/baseline" \
            --batch-size 32 \
            --k 5

        for num_styles in "${NUM_STYLES_TO_CHANGE[@]}"; do
            PIPELINE_SUFFIX="promptremix-authbench-qwen3.6-${num_styles}"
            python scripts/7-baselines/authbench/authbench_eval.py \
                --model "${model}" \
                --language "${LANGUAGE}" \
                --genre "${genre}" \
                --pipeline-suffix "${PIPELINE_SUFFIX}" \
                --output-dir "${OUTPUT_DIR}/${LANGUAGE}/${PIPELINE_SUFFIX}" \
                --batch-size 32 \
                --k 5 || echo "Skipping ${genre}/${PIPELINE_SUFFIX}"
        done
    done
done
