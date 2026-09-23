#!/bin/bash
# One short-text jamdec genre. Set LANGUAGE and GENRE via --export.
#SBATCH --job-name=short_jamdec_g
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
#SBATCH --export=ALL
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/short_jamdec_%x_%j.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/short_jamdec_%x_%j.err
set -e

if [[ -z "${LANGUAGE:-}" || -z "${GENRE:-}" ]]; then
    echo "LANGUAGE and GENRE must be set (sbatch --export=ALL,LANGUAGE=en,GENRE=poetry ...)" >&2
    exit 1
fi

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0 coenv/jdk/17.0.8

SAMPLE_ROOT="/gscratch/stf/hardiksr/hiatus/data/sampled/authbench_short_text/bottom25"
PRIVACY_ROOT="/gscratch/stf/hardiksr/hiatus/metrics/paper_experiments/short_text/baselines/jamdec/privacy/${LANGUAGE}"
SENSE_ROOT="/gscratch/stf/hardiksr/hiatus/metrics/paper_experiments/short_text/baselines/jamdec/sense/${LANGUAGE}"
mkdir -p "${PRIVACY_ROOT}" "${SENSE_ROOT}"

suffix="obfuscate_eval_jamdec_authbench_${LANGUAGE}-baseline-short-bottom25"
sample="${SAMPLE_ROOT}/${LANGUAGE}/${GENRE}/authbench_${LANGUAGE}_${GENRE}_sampled_ids.json"
if [[ ! -s "${sample}" ]]; then
    echo "Missing short-text sample: ${sample}" >&2
    exit 1
fi

echo "Short-text jamdec LANGUAGE=${LANGUAGE} GENRE=${GENRE}"

read -r original privatized _ _ < <(
    python scripts/7-baselines/authbench/get_authbench_paths.py \
        --language "${LANGUAGE}" \
        --genre "${GENRE}" \
        --pipeline-suffix "${suffix}"
)

expected=$(python3 -c "import json; print(len(json.load(open('${sample}'))))")
actual=0
if [[ -f "${privatized}" ]]; then
    actual=$(grep -c . "${privatized}" || true)
fi

if [[ "${actual}" -eq "${expected}" ]]; then
    echo "Skipping obfuscate ${LANGUAGE}/${GENRE} (complete: ${actual}/${expected})"
else
    echo "Obfuscating ${LANGUAGE}/${GENRE} (${actual}/${expected} done, will resume)"
    python scripts/7-baselines/jamdec/obfuscate_jamdec.py \
        --dataset authbench \
        --language "${LANGUAGE}" \
        --genre "${GENRE}" \
        --output-suffix "${suffix}" \
        --sampled-data-filepath "${sample}" \
        --cache-dir "/gscratch/stf/hardiksr/hiatus/cache"
    actual=$(grep -c . "${privatized}" || true)
fi

if [[ "${actual}" -ne "${expected}" ]]; then
    echo "Obfuscation incomplete for ${LANGUAGE}/${GENRE}: ${actual}/${expected}" >&2
    exit 1
fi

python scripts/7-baselines/authbench/authbench_eval.py \
    --model "Blablablab/multilingual-style-representation" \
    --language "${LANGUAGE}" \
    --dataset authbench \
    --genre "${GENRE}" \
    --pipeline-suffix "${suffix}" \
    --output-dir "${PRIVACY_ROOT}" \
    --batch-size 32 \
    --sampled-data-filepath "${sample}"

sense="${SENSE_ROOT}/${GENRE}_${suffix}_llm.jsonl"
if [[ ! -s "${sense}" ]]; then
    python scripts/5-evaluation/score_sense.py \
        --text-original-jsonl "${original}" \
        --text-privatized-jsonl "${privatized}" \
        --output-path "${sense}" \
        --model-path "mistralai/Mistral-Small-24B-Instruct-2501" \
        --batch-size 16 \
        --sampled-data-filepath "${sample}"
fi

echo "Done ${LANGUAGE}/${GENRE}"
