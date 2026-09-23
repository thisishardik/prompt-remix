#!/bin/bash

#SBATCH --job-name=download_blog_data
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --mem=3GB
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/download_blog_data.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/download_blog_data.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm

OUTPUT_DIR="/gscratch/ifml1/hiatus/data/raw/blog_authorship/sample_2000"

python scripts/7-baselines/blog_authorship/download_blog_authorship.py \
    --output_dir "$OUTPUT_DIR" \
    --num_authors 2000 \
    --queries_per_author 1 \
    --candidates_per_author 10 \
    --seed 42
