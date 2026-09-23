#!/bin/bash

#SBATCH --job-name=authbench_eval-zh
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=75GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=80:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/authbench_eval-zh.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/authbench_eval-zh.err
set -e

source /mmfs1/home/hardiksr/.bashrc
module load cuda gcc

num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}

language="zh"
zh_genres=("perGenre-HRS3.301" "perGenre-HRS3.302" "perGenre-HRS3.303" "perGenre-HRS3.304" "perGenre-HRS3.305" "zh_short_crossGenre" "zh_medium_crossGenre")
models=("deepseek-ai/deepseek-coder-6.7b-instruct" "Qwen/Qwen3-4B" "Qwen/Qwen3-Embedding-8B" "intfloat/multilingual-e5-large")

for model in "${models[@]}"; do
    echo "Running evaluation with: ${model}"

    for num_styles in "${num_styles_to_change[@]}"; do
        pipeline_suffix="promptremix-zh0-qwen3.5-${num_styles}"

        for genre in "${zh_genres[@]}"; do
            echo "Running evaluation for genre: ${genre}"

            conda activate vllm
            read -r input_queries_path privatized_queries_path candidate_queries_path ground_truth_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language ${language} --pipeline-suffix ${pipeline_suffix})

            output_dir="/gscratch/stf/hardiksr/hiatus/metrics/authbench/${language}/${genre}/${pipeline_suffix}"

            python scripts/7-baselines/authbench/authbench_baseline.py \
                --model "${model}" \
                --queries-jsonl "${input_queries_path}" \
                --candidates-jsonl "${candidate_queries_path}" \
                --privatized-queries-jsonl "${privatized_queries_path}" \
                --ground-truth-file "${ground_truth_path}" \
                --output-dir "${output_dir}" \
                --batch-size 32
        done
    done
done