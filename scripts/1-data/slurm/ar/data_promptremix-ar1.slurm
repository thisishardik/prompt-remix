#!/bin/bash

#SBATCH --job-name=data_promptremix-ar1
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --gpus=1
#SBATCH --time=10:00:00 
#SBATCH --cpus-per-task=6
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/data_promptremix-ar1.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/data_promptremix-ar1.err

source /mmfs1/home/hardiksr/.bashrc
cd /mmfs1/home/hardiksr/repository/hiatus/scripts
conda activate hiatus
module load cuda gcc

#python 1-data/download_wikipedia_data.py \
#    --language "zh" \
#    --date 20250201 \
#    --n-samples 4000 \
#    --output-path /gscratch/stf/mpotto/hiatus/data/zh/raw/wiki_zh0.jsonl

python 1-data/split_data_classifiers_adapters.py \
    --input-path /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar.json \
    --adapter-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar1_adapters.jsonl \
    --classifier-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar1_classifiers.jsonl \
    --n-samples 3000