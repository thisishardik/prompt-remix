#!/bin/bash

#SBATCH --job-name=blog_eval_en
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
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/blog_eval_en.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/blog_eval_en.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda gcc

LANGUAGE="en"
DATASET="blog"
SAMPLE_NAME="sample_2000"
PIPELINE_SUFFIX="promptremix-blog-qwen3.6-3"
OUTPUT_DIR="/gscratch/stf/hardiksr/hiatus/metrics/blog_authorship"
MODELS=("Qwen/Qwen2.5-3B-Instruct")

for model in "${MODELS[@]}"; do
    echo "Model: ${model}, Dataset: ${DATASET}, Sample: ${SAMPLE_NAME}"

    python scripts/7-baselines/authbench/authbench_eval.py \
        --model "${model}" \
        --language "${LANGUAGE}" \
        --dataset "${DATASET}" \
        --sample-name "${SAMPLE_NAME}" \
        --pipeline-suffix "${PIPELINE_SUFFIX}" \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size 32 \
        --k 5
done
