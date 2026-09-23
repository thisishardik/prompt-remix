#!/bin/bash
#SBATCH --job-name=pr_random_axis_ru_eval
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=90GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=80:00:00
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/pr_random_axis_ru_eval.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/pr_random_axis_ru_eval.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0 coenv/jdk/17.0.8

LANGUAGE="ru"
SAMPLE_ROOT="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench"
PRIVACY_ROOT="/gscratch/stf/hardiksr/hiatus/metrics/paper_experiments/random_axis/privacy/${LANGUAGE}"
SENSE_ROOT="/gscratch/stf/hardiksr/hiatus/metrics/paper_experiments/random_axis/sense/${LANGUAGE}"
FORCE_SENSE=1
genres=(
    "literature"
    "news"
    "poetry"
    "qna"
    "social_media"
)
suffixes=(
    "promptremix_authbench-ru0-3"
    "promptremix_authbench-ru0-3-random-seed42"
    "promptremix_authbench-ru0-3-random-seed43"
    "promptremix_authbench-ru0-3-random-seed44"
    "promptremix_authbench-ru0-3-random-seed45"
    "promptremix_authbench-ru0-3-random-seed46"
)

mkdir -p "${PRIVACY_ROOT}" "${SENSE_ROOT}"
for suffix in "${suffixes[@]}"; do
    for genre in "${genres[@]}"; do
        sample="${SAMPLE_ROOT}/${LANGUAGE}/${genre}/authbench_${LANGUAGE}_${genre}_sampled_ids.json"
        read -r original privatized _ _ < <(
            python scripts/7-baselines/authbench/get_authbench_paths.py \
                --language "${LANGUAGE}" \
                --genre "${genre}" \
                --pipeline-suffix "${suffix}"
        )
        if [[ ! -s "${sample}" || ! -s "${privatized}" ]]; then
            echo "Missing sample or privatized output for ${genre}/${suffix}" >&2
            continue
        fi

        python scripts/7-baselines/authbench/authbench_eval.py \
            --model "Blablablab/multilingual-style-representation" \
            --language "${LANGUAGE}" \
            --dataset authbench \
            --genre "${genre}" \
            --pipeline-suffix "${suffix}" \
            --output-dir "${PRIVACY_ROOT}" \
            --batch-size 32 \
            --sampled-data-filepath "${sample}"

        sense="${SENSE_ROOT}/${genre}_${suffix}_llm.jsonl"
        if [[ -s "${sense}" && "${FORCE_SENSE}" != "1" ]]; then
            echo "Sense output exists, skipping: ${sense}"
        else
            python scripts/5-evaluation/score_sense.py \
                --text-original-jsonl "${original}" \
                --text-privatized-jsonl "${privatized}" \
                --output-path "${sense}" \
                --model-path "mistralai/Mistral-Small-24B-Instruct-2501" \
                --batch-size 16 \
                --sampled-data-filepath "${sample}"
        fi
    done
done
