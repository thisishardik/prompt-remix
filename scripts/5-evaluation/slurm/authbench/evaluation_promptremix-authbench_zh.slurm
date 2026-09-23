#!/bin/bash

#SBATCH --job-name=evaluation_promptremix-autbench_zh
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
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/evaluation_promptremix-autbench_zh.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/evaluation_promptremix-autbench_zh.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda gcc coenv/jdk/17.0.8

num_styles_to_change=(1 2 3)
genres=("ecommerce_reviews" "literature" "media_reviews" "news" "poetry" "social_media")
MODEL="Qwen/Qwen2.5-3B-Instruct"
METRICS_DIR="/gscratch/stf/hardiksr/hiatus/metrics/promptremix"
language="zh"

for num_styles in "${num_styles_to_change[@]}"; do
    pipeline_suffix="promptremix_authbench-zh0-${num_styles}"

    for genre in "${genres[@]}"; do
        echo "Running evaluation for genre: ${genre}"

        sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${language}/${genre}/authbench_${language}_${genre}_sampled_ids.json" 
        
        # --- Privacy and Stylistic Consistency ---
        conda activate hiatus-eval
        python scripts/5-evaluation/score_privacy_consistency.py \
            --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
            --genre "$genre" \
            --language "zh" \
            --pipeline-suffix ${pipeline_suffix} \
            --batch-size 32 \
            --sampled-data-filepath ${sampled_data_filepath}

        # --- AuthBench Eval ---
        echo "Running AuthBench Delta EER Eval for PromptRemix (${language}, ${genre})"
        conda activate vllm
        python scripts/7-baselines/authbench/authbench_eval.py \
            --model "${MODEL}" \
            --language "${language}" \
            --dataset authbench \
            --genre "${genre}" \
            --pipeline-suffix "${pipeline_suffix}" \
            --output-dir "${METRICS_DIR}/authbench" \
            --batch-size 32 \
            --sampled-data-filepath "${sampled_data_filepath}"

        # --- Sense Metrics ---
        conda activate vllm
        read -r input_queries_path privatized_queries_path _ _ < <(python scripts/7-baselines/authbench/get_authbench_paths.py --genre ${genre} --language "zh" --pipeline-suffix ${pipeline_suffix})

        output_path_llm="/gscratch/stf/hardiksr/hiatus/metrics/zh/${genre}_${pipeline_suffix}_llm.jsonl"

        python scripts/5-evaluation/score_sense.py \
            --text-original-jsonl ${input_queries_path} \
            --text-privatized-jsonl ${privatized_queries_path} \
            --output-path ${output_path_llm} \
            --model-path "mistralai/Mistral-Small-24B-Instruct-2501" \
            --batch-size 16 \
            --sampled-data-filepath ${sampled_data_filepath}
    done
done

echo "Accumulating Delta EER Metrics"
conda activate vllm

METHODS=()
for num_styles in "${num_styles_to_change[@]}"; do
    METHODS+=("promptremix_authbench-${language}0-${num_styles}")
done

python scripts/5-evaluation/accumulate_delta_eer.py \
    --method "${METHODS[@]}" \
    --group-name "promptremix_authbench_${language}-baseline" \
    --dataset authbench \
    --language "${language}" \
    --model-tag "${MODEL//\//_}" \
    --genres "${genres[@]}" \
    --baseline-dir "${METRICS_DIR}" \
    --authbench-dir "${METRICS_DIR}/authbench" \
    --hiatus-dir "/gscratch/stf/hardiksr/hiatus/metrics/${language}" \
    --out-dir "/gscratch/stf/hardiksr/hiatus/metrics"
