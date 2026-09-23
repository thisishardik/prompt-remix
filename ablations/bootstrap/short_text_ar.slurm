#!/bin/bash
#SBATCH --job-name=pr_short_text_ar_boot
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
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/pr_short_text_ar_boot.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/pr_short_text_ar_boot.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0 coenv/jdk/17.0.8

LANGUAGE="ar"
PRIVACY_ROOT="/gscratch/stf/hardiksr/hiatus/metrics/paper_experiments/short_text/privacy/ar"
OUTPUT_ROOT="/gscratch/stf/hardiksr/hiatus/metrics/paper_experiments/short_text/bootstrap/${LANGUAGE}"
genres=(
    "literature"
    "news"
    "poetry"
    "social_media"
)
suffixes=(
    "promptremix_authbench-ar0-1-short-bottom25"
    "promptremix_authbench-ar0-2-short-bottom25"
    "promptremix_authbench-ar0-3-short-bottom25"
)

mkdir -p "${OUTPUT_ROOT}"
for suffix in "${suffixes[@]}"; do
    for genre in "${genres[@]}"; do
        embeddings="${PRIVACY_ROOT}/embeddings_authbench_${LANGUAGE}_${genre}_Blablablab_multilingual-style-representation_${suffix}.npz"
        if [[ ! -s "${embeddings}" ]]; then
            echo "Missing bootstrap embeddings: ${embeddings}" >&2
            continue
        fi
        python ablations/bootstrap/bootstrap_delta_eer.py \
            --embeddings "${embeddings}" \
            --output-prefix "${OUTPUT_ROOT}/${genre}_${suffix}" \
            --n-bootstrap 1000 \
            --workers 6 \
            --seed 42
    done
done
