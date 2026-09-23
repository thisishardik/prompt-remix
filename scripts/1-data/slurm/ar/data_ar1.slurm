#!/bin/bash

#SBATCH --job-name=data_ar
#SBATCH --mail-user=mpotto@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=0
#SBATCH --cpus-per-task=10
#SBATCH --time=8:00:00 

#SBATCH --chdir=/mmfs1/home/mpotto/code/hiatus/scripts
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/mpotto/hiatus/logs/data_ar.out
#SBATCH --error=/gscratch/stf/mpotto/hiatus/logs/data_ar.err
set -e

source /mmfs1/home/mpotto/.bashrc
conda activate hiatus
module load cuda gcc
module load coenv/jdk/17.0.8

python 1-data/download_wikipedia_data.py \
    --language "ar" \
    --date 20251101 \
    --n-samples 100 \
    --num-workers 8 \
    --output-path /gscratch/stf/mpotto/hiatus/data/ar/raw/wiki_ar.jsonl

# # Mini
# python 1-data/split_data_classifiers_adapters.py \
#     --n-samples 100 \
#     --input-path /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar.jsonl \
#     --adapter-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar_mini_adapters.jsonl \
#     --classifier-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar_mini_classifiers.jsonl 

# # Large
# python 1-data/split_data_classifiers_adapters.py \
#     --n-samples 2500 \
#     --input-path /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar.jsonl \
#     --adapter-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar_large_adapters.jsonl \
#     --classifier-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar_large_classifiers.jsonl 


# # Huge
# python 1-data/split_data_classifiers_adapters.py \
#     --n-samples 5000  \
#     --input-path /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar.jsonl \
#     --adapter-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar_huge_adapters.jsonl \
#     --classifier-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar_huge_classifiers.jsonl 
