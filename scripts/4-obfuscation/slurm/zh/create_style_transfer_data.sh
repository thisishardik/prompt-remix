#!/bin/bash

#SBATCH --job-name=obfuscation_promptremix-tfs_zh0-transfer-qwen3_5
#SBATCH --mail-user=kogolobo@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=50:00:00 

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-tfs_zh0-transfer-qwen3_5.out
#SBATCH --error=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_promptremix-tfs_zh0-transfer-qwen3_5.err
set -e

source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda/12.8.1 gcc/12.3.0

# Genres for Chinese
zh_genres=("perGenre-HRS3.301" "perGenre-HRS3.302" "perGenre-HRS3.303" "perGenre-HRS3.304" "perGenre-HRS3.305" "zh_short_crossGenre" "zh_medium_crossGenre")

# Create style transfer data
for genre in "${zh_genres[@]}"; do
    read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language "zh" --pipeline-suffix test)
    mkdir -p "/gscratch/ifml1/kogolobo/style_transfer/zh/${genre}"
    output_path="/gscratch/ifml1/kogolobo/style_transfer/zh/${genre}/target_author_data.json"
    echo "Creating style transfer data for genre ${genre}"
    echo "Input path: ${input_queries_path}"
    echo "Output path: ${output_path}"

    python scripts/4-obfuscation/make_style_transfer_data.py \
        --input-path ${input_queries_path} \
        --output-path ${output_path} \
        --input-key "fullText" \
        --output-key "PIIfullText" \
        --seed 42
done