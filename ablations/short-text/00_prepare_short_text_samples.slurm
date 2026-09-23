#!/bin/bash
#SBATCH --job-name=pr_short_prepare
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/pr_short_prepare.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/pr_short_prepare.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm

python scripts/5-evaluation/build_authbench_short_text_manifest.py \
    --length-source attributor \
    --output-dir promptremix-paper/short-text
python ablations/short-text/prepare_short_text_samples.py \
    --manifest promptremix-paper/short-text/short_text_manifest.jsonl \
    --authors-per-cell 100 \
    --max-docs-per-author 2 \
    --seed 42
