#!/bin/bash

#SBATCH --job-name=evaluate_blabla_ar
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=90GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=80:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/evaluate_blabla_ar.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/evaluate_blabla_ar.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0

LANGUAGE="ar"
MODEL="Blablablab/multilingual-style-representation"

genres=(
    "literature"
    "news"
    "poetry"
    "social_media"
)

# List of all pipeline suffixes (7 baselines + promptremix)
METHODS=(
    # "obfuscate_eval_jamdec_authbench_${LANGUAGE}-baseline"
    "mutantx"
    "steer"
    "paraphrasing"
    "round_trip_mt"
    "stylometry"
    "obfuscate_eval_styleremix_authbench_${LANGUAGE}-baseline-1"
    "obfuscate_eval_styleremix_authbench_${LANGUAGE}-baseline-2"
    "obfuscate_eval_styleremix_authbench_${LANGUAGE}-baseline-3"
    "promptremix_authbench-${LANGUAGE}0-1"
    "promptremix_authbench-${LANGUAGE}0-2"
    "promptremix_authbench-${LANGUAGE}0-3"
)

get_method_name() {
    local suffix=$1
    if [[ $suffix == *"jamdec"* ]]; then echo "jamdec"
    elif [[ $suffix == *"mutantx"* ]]; then echo "mutantx"
    elif [[ $suffix == *"steer"* ]]; then echo "steer"
    elif [[ $suffix == *"paraphrasing"* ]]; then echo "paraphrasing"
    elif [[ $suffix == *"round_trip_mt"* ]]; then echo "round_trip_mt"
    elif [[ $suffix == *"stylometry"* ]]; then echo "stylometry"
    elif [[ $suffix == *"styleremix"* ]]; then echo "styleremix"
    elif [[ $suffix == *"promptremix"* ]]; then echo "promptremix"
    else echo "other"
    fi
}

for method in "${METHODS[@]}"; do
    echo "======================================"
    echo "Evaluating method: $method"
    echo "======================================"

    METHOD_NAME=$(get_method_name "$method")
    METRICS_DIR="/gscratch/stf/hardiksr/hiatus/metrics/${METHOD_NAME}/authbench_blabla"

    for genre in "${genres[@]}"; do
        echo "Running AuthBench Delta EER Eval for ($LANGUAGE, $genre, $method)"
        sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${genre}/authbench_${LANGUAGE}_${genre}_sampled_ids.json"

        python scripts/7-baselines/authbench/authbench_eval.py \
            --model "${MODEL}" \
            --language "${LANGUAGE}" \
            --dataset authbench \
            --genre "${genre}" \
            --pipeline-suffix "${method}" \
            --output-dir "${METRICS_DIR}/authbench" \
            --batch-size 32 \
            --sampled-data-filepath "${sampled_data_filepath}" || echo "Failed $method"
    done

    echo "Accumulating Delta EER Metrics for $method"
    python scripts/5-evaluation/accumulate_delta_eer.py \
        --method "${method}" \
        --group-name "${method}_${LANGUAGE}" \
        --dataset authbench \
        --language "${LANGUAGE}" \
        --model-tag "${MODEL//\//_}" \
        --genres "${genres[@]}" \
        --baseline-dir "${METRICS_DIR}" \
        --authbench-dir "${METRICS_DIR}/authbench" \
        --hiatus-dir "/gscratch/stf/hardiksr/hiatus/metrics/${LANGUAGE}" \
        --out-dir "/gscratch/stf/hardiksr/hiatus/metrics" || echo "Failed accumulate $method"
done

echo "Done evaluating all methods for AR."
