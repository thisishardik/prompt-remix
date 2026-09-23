#!/bin/bash
#SBATCH --job-name=short_styleremix_%a
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=90GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=80:00:00
#SBATCH --array=0-3
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/short_styleremix_%A_%a.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/short_styleremix_%A_%a.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0 coenv/jdk/17.0.8

languages=(en ru zh ar)
LANGUAGE="${languages[$SLURM_ARRAY_TASK_ID]}"
case "${LANGUAGE}" in
    en)
        genres=(blog ecommerce_reviews literature news poetry qna research_paper)
        CONFIG_FILE="scripts/4-obfuscation/config/remix_config_en.json"
        ;;
    ru)
        genres=(literature news poetry qna social_media)
        CONFIG_FILE="scripts/4-obfuscation/config/remix_config_ru.json"
        ;;
    zh)
        genres=(ecommerce_reviews literature media_reviews news poetry social_media)
        CONFIG_FILE="scripts/4-obfuscation/config/remix_config_zh.json"
        ;;
    ar)
        genres=(literature news poetry social_media)
        CONFIG_FILE="scripts/4-obfuscation/config/remix_config_ar3.json"
        ;;
esac

SAMPLE_ROOT="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench_short_text/bottom25"
PRIVACY_ROOT="/gscratch/stf/hardiksr/hiatus/metrics/paper_experiments/short_text/baselines/styleremix/privacy/${LANGUAGE}"
SENSE_ROOT="/gscratch/stf/hardiksr/hiatus/metrics/paper_experiments/short_text/baselines/styleremix/sense/${LANGUAGE}"
mkdir -p "${PRIVACY_ROOT}" "${SENSE_ROOT}"
extra_args=(

)

num_styles_values=(1 2 3)
for num_styles in "${num_styles_values[@]}"; do
    suffix="obfuscate_eval_styleremix_authbench_${LANGUAGE}-baseline-${num_styles}-short-bottom25"
    style_args=(--top-n-styles-to-change "${num_styles}" --config-file "${CONFIG_FILE}")
    for genre in "${genres[@]}"; do
        sample="${SAMPLE_ROOT}/${LANGUAGE}/${genre}/authbench_${LANGUAGE}_${genre}_sampled_ids.json"
        if [[ ! -s "${sample}" ]]; then
            echo "Missing short-text sample: ${sample}" >&2
            continue
        fi

        read -r original privatized _ _ < <(
            python scripts/7-baselines/authbench/get_authbench_paths.py \
                --language "${LANGUAGE}" \
                --genre "${genre}" \
                --pipeline-suffix "${suffix}"
        )
        if [[ ! -s "${privatized}" ]]; then
            python "scripts/7-baselines/styleremix/obfuscate_styleremix.py" \
                --dataset authbench \
                --language "${LANGUAGE}" \
                --genre "${genre}" \
                --output-suffix "${suffix}" \
                --sampled-data-filepath "${sample}" \
            "${style_args[@]}" \
                "${extra_args[@]}"
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
        if [[ ! -s "${sense}" ]]; then
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
