#!/bin/bash

#SBATCH --job-name=data_promptremix-tfs_ar3
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --gpus=1
#SBATCH --time=10:00:00 
#SBATCH --cpus-per-task=6
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/data_promptremix-tfs_ar3.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/data_promptremix-tfs_ar3.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate hiatus
module load cuda gcc
module load coenv/jdk/17.0.8

python scripts/1-data/download_wikipedia_data.py \
   --language "ar" \
   --date 20251101 \
   --n-samples 5000 \
   --output-path /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar3.jsonl

python scripts/1-data/split_data_classifiers_adapters.py \
    --input-path /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar3.jsonl \
    --adapter-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar3_adapters.jsonl \
    --classifier-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar3_classifiers.jsonl \
    --n-samples 5000