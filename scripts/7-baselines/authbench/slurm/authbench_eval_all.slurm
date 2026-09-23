#!/bin/bash

#SBATCH --job-name=authbench_eval_all
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=80:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/authbench_eval_all.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/authbench_eval_all.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda gcc

LANGUAGES=("en" "ru" "zh" "ar")
SPLIT="test"
OUTPUT_DIR="/gscratch/stf/hardiksr/hiatus/metrics/authbench"

MODELS=("meta-llama/Meta-Llama-3-8B" "Qwen/Qwen2.5-3B-Instruct")

NUM_STYLES_TO_CHANGE=(1 2 3)

for model in "${MODELS[@]}"; do
    echo "Evaluating with model: ${model}"

    for lang in "${LANGUAGES[@]}"; do
        echo "Language: ${lang}"

        echo "Running baseline evaluation (no obfuscation)..."
        python scripts/7-baselines/authbench/authbench_eval.py \
            --model "${model}" \
            --language "${lang}" \
            --split "${SPLIT}" \
            --output-dir "${OUTPUT_DIR}/${lang}/baseline" \
            --batch-size 32 \
            --k 5

        for num_styles in "${NUM_STYLES_TO_CHANGE[@]}"; do
            PIPELINE_SUFFIX="promptremix-authbench-qwen3.6-${num_styles}"

            echo "Evaluating with pipeline: ${PIPELINE_SUFFIX}..."
            python scripts/7-baselines/authbench/authbench_eval.py \
                --model "${model}" \
                --language "${lang}" \
                --split "${SPLIT}" \
                --pipeline-suffix "${PIPELINE_SUFFIX}" \
                --output-dir "${OUTPUT_DIR}/${lang}/${PIPELINE_SUFFIX}" \
                --batch-size 32 \
                --k 5 || echo "Skipping ${lang}/${PIPELINE_SUFFIX} (privatized queries not found)"
        done
    done
done
