#!/bin/bash

#SBATCH --job-name=sense_styleremix_ar
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
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/sense_styleremix_ar.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/sense_styleremix_ar.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0 coenv/jdk/17.0.8

LANGUAGE="ar"
SENSE_MODEL="mistralai/Mistral-Small-24B-Instruct-2501"
METRICS_ROOT="/gscratch/stf/hardiksr/hiatus/metrics"
SAMPLED_ROOT="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench"

genres=(
    "literature"
    "news"
    "poetry"
    "social_media"
)

num_styles_to_change=(
    1
    2
    3
)

for num_styles in "${num_styles_to_change[@]}"; do
    pipeline_suffix="obfuscate_eval_styleremix_authbench_ar-baseline-${num_styles}"
    echo "Scoring StyleRemix with ${num_styles} style(s)."
    for genre in "${genres[@]}"; do
        sampled_data_filepath="${SAMPLED_ROOT}/${LANGUAGE}/${genre}/authbench_${LANGUAGE}_${genre}_sampled_ids.json"
        output_path_llm="${METRICS_ROOT}/${LANGUAGE}/${genre}_${pipeline_suffix}_llm.jsonl"

        if [[ ! -s "${sampled_data_filepath}" ]]; then
            echo "Missing sampled IDs: ${sampled_data_filepath}" >&2
            continue
        fi

        read -r input_queries_path privatized_queries_path _ _ < <(
            python scripts/7-baselines/authbench/get_authbench_paths.py \
                --genre "${genre}" \
                --language "${LANGUAGE}" \
                --pipeline-suffix "${pipeline_suffix}"
        )

        if [[ ! -s "${input_queries_path}" ]]; then
            echo "Missing original queries: ${input_queries_path}" >&2
            continue
        fi
        if [[ ! -s "${privatized_queries_path}" ]]; then
            echo "Missing privatized queries: ${privatized_queries_path}" >&2
            continue
        fi
        if [[ -s "${output_path_llm}" ]]; then
            echo "Sense scores already exist, skipping: ${output_path_llm}"
            continue
        fi

        echo "Scoring ${LANGUAGE}/${genre} with ${pipeline_suffix}"
        python scripts/5-evaluation/score_sense.py \
            --text-original-jsonl "${input_queries_path}" \
            --text-privatized-jsonl "${privatized_queries_path}" \
            --output-path "${output_path_llm}" \
            --model-path "${SENSE_MODEL}" \
            --batch-size 16 \
            --sampled-data-filepath "${sampled_data_filepath}"
    done
done

echo "Finished sense scoring for styleremix/ar."
