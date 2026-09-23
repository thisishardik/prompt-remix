#!/bin/bash

#SBATCH --job-name=obfuscate_eval_jamdec_authbench_ru
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=90GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=5
#SBATCH --time=80:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_jamdec_authbench_ru.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_jamdec_authbench_ru.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0

LANGUAGE="ru"
MODEL="meta-llama/Meta-Llama-3-8B"
PIPELINE_SUFFIX="obfuscate_eval_jamdec_authbench_ru-baseline"
METRICS_DIR="/gscratch/stf/hardiksr/hiatus/metrics/jamdec"
CACHE_DIR="/gscratch/stf/hardiksr/hiatus/cache"

genres=(
    "literature"
    "news"
    "poetry"
    "qna"
    "social_media"
)

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

genre_obfuscation_complete() {
    local genre="$1"
    local privatized_path="/gscratch/stf/hardiksr/hiatus/privatized/authbench/${LANGUAGE}/authbench_${LANGUAGE}_${genre}_privatized_queries_${PIPELINE_SUFFIX}.jsonl"
    local sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${genre}/authbench_${LANGUAGE}_${genre}_sampled_ids.json"
    if [[ ! -f "${privatized_path}" || ! -f "${sampled_data_filepath}" ]]; then
        return 1
    fi
    local expected actual
    expected=$(python3 -c "import json; print(len(json.load(open('${sampled_data_filepath}'))))")
    actual=$(wc -l < "${privatized_path}")
    [[ "${actual}" -eq "${expected}" ]]
}

for GENRE in "${genres[@]}"; do
    privatized_path="/gscratch/stf/hardiksr/hiatus/privatized/authbench/${LANGUAGE}/authbench_${LANGUAGE}_${GENRE}_privatized_queries_${PIPELINE_SUFFIX}.jsonl"
    if genre_obfuscation_complete "${GENRE}"; then
        echo "Skipping obfuscate genre ${GENRE} (complete: ${privatized_path})"
        continue
    fi
    if [[ -f "${privatized_path}" ]]; then
        echo "Removing partial privatized output for ${GENRE}: ${privatized_path}"
        rm -f "${privatized_path}"
    fi
    echo "Obfuscating genre: ${GENRE}"
    sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${GENRE}/authbench_${LANGUAGE}_${GENRE}_sampled_ids.json"

    python scripts/7-baselines/jamdec/obfuscate_jamdec.py \
        --dataset authbench \
        --language "${LANGUAGE}" \
        --genre "${GENRE}" \
        --output-suffix "${PIPELINE_SUFFIX}" \
        --cache-dir "${CACHE_DIR}" \
        --sampled-data-filepath "${sampled_data_filepath}" \

done

for GENRE in "${genres[@]}"; do
    echo "Evaluating genre: ${GENRE}"
    sampled_data_filepath="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench/${LANGUAGE}/${GENRE}/authbench_${LANGUAGE}_${GENRE}_sampled_ids.json"

    # AuthBench Eval
    echo "Running AuthBench Delta EER Eval for JAMDEC (${LANGUAGE}, ${GENRE})"
    conda activate vllm
    python scripts/7-baselines/authbench/authbench_eval.py \
        --model "${MODEL}" \
        --language "${LANGUAGE}" \
        --dataset authbench \
        --genre "${GENRE}" \
        --pipeline-suffix "${PIPELINE_SUFFIX}" \
        --output-dir "${METRICS_DIR}/authbench" \
        --batch-size 32 \
        --sampled-data-filepath "${sampled_data_filepath}"

    # Hiatus Eval
    echo "Running Hiatus Delta EER Eval for JAMDEC (${LANGUAGE}, ${GENRE})"
    conda activate hiatus-eval
    python scripts/5-evaluation/score_privacy_consistency.py \
        --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
        --genre "${GENRE}" \
        --language "${LANGUAGE}" \
        --pipeline-suffix "${PIPELINE_SUFFIX}" \
        --batch-size 32 \
        --sampled-data-filepath "${sampled_data_filepath}"
done

echo "Accumulating Delta EER Metrics"

conda activate vllm
python scripts/5-evaluation/accumulate_delta_eer.py \
    --method "${PIPELINE_SUFFIX}" \
    --group-name "obfuscate_eval_jamdec_authbench_ru-baseline" \
    --dataset authbench \
    --language "${LANGUAGE}" \
    --model-tag "${MODEL//\//_}" \
    --genres "${genres[@]}" \
    --baseline-dir "${METRICS_DIR}" \
    --authbench-dir "${METRICS_DIR}/authbench" \
    --hiatus-dir "/gscratch/stf/hardiksr/hiatus/metrics/${LANGUAGE}" \
    --out-dir "/gscratch/stf/hardiksr/hiatus/metrics"
