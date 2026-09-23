#!/bin/bash
#SBATCH --job-name=pr_short_text_k2_ar_obf
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
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/pr_short_text_k2_ar_obf.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/pr_short_text_k2_ar_obf.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0 coenv/jdk/17.0.8

LANGUAGE="ar"
PIPELINE_SUFFIX="promptremix_authbench-ar0-2-short-bottom25"
SAMPLE_ROOT="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench_short_text/bottom25"
FORCE_OBF="${FORCE_OBF:-0}"
genres=(
    "literature"
    "news"
    "poetry"
    "social_media"
)

for genre in "${genres[@]}"; do
    sample="${SAMPLE_ROOT}/${LANGUAGE}/${genre}/authbench_${LANGUAGE}_${genre}_sampled_ids.json"
    if [[ ! -s "${sample}" ]]; then
        echo "Missing sample: ${sample}" >&2
        continue
    fi
    read -r original privatized _ _ < <(
        python scripts/7-baselines/authbench/get_authbench_paths.py \
            --language "${LANGUAGE}" \
            --genre "${genre}" \
            --pipeline-suffix "${PIPELINE_SUFFIX}"
    )
    if [[ -s "${privatized}" && "${FORCE_OBF}" != "1" ]]; then
        echo "Output exists, skipping: ${privatized}"
        continue
    fi

    tmp="/gscratch/stf/hardiksr/hiatus/scrubbed/short_text_k2_${LANGUAGE}_${genre}_${PIPELINE_SUFFIX}.jsonl"
    python scripts/4-obfuscation/insert_pii.py \
        --input-path "${original}" \
        --input-key fullText \
        --lang "${LANGUAGE}" \
        --name "نور" \
        --output-key PIIfullText \
        --output-path "${tmp}" \
        --sampled-data-filepath "${sample}"

    python scripts/4-obfuscation/run_remix.py \
        --input-file "${tmp}" \
        --model-id "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" \
        --prompt-remix \
        --lang "${LANGUAGE}" \
            --library "corenlp" \
        --config-file "scripts/4-obfuscation/config/remix_config_ar3.json" \
        --text-key PIIfullText \
        --author-key authorIDs \
        --document-key documentID \
        --top-n-styles-to-change 2 \
        --direction-seed 42 \
        --generation-seed 42 \
        --output-key PIIobfuscated \
        --max-gpu-memory-utilization 0.60 \
        --output-file "${tmp}"

    python scripts/4-obfuscation/remove_pii.py \
        --input-path "${tmp}" \
        --input-key PIIobfuscated \
        --output-key PIIobfuscatedPostEditNoPII \
        --output-path "${privatized}" \
        --lang "${LANGUAGE}" \
        --name "نور"
    python scripts/4-obfuscation/rename_columns.py \
        --input-path "${privatized}" \
        --output-path "${privatized}" \
        --old-key PIIobfuscatedPostEditNoPII \
        --new-key fullText
done
