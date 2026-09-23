#!/bin/bash

#SBATCH --job-name=paraphrase_jais
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=ALL
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=5
#SBATCH --time=50:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus-dev
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/out/ar_paraphrase_jais_ifml1.txt # where STDOUT goes
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/err/ar_paraphrase_jais_ifml1.txt # where STDERR goes

source /mmfs1/home/hardiksr/.bashrc
conda activate hiatus
module load cuda gcc

version="v0"
output_dir="/gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/$version/ar_paraphrases_jais"
prompts=$(ls prompts/prompts_ar/*.j2)

for prompt in $prompts
do
    echo "Processing $prompt"
    python scripts/2-paraphrase/paraphrase_qwen.py \
        --prompts $prompt \
        --model_name inceptionai/jais-adapted-7b-chat \
        --classifier_base_texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/ar_samples_classifier.jsonl \
        --output_dir $output_dir

done


# Bookkeeping to track metrics

bookkeeping_out_dir="/gscratch/stf/hardiksr/hiatus/bookkeeping/$version/out/"
bookkeeping_err_dir="/gscratch/stf/hardiksr/hiatus/bookkeeping/$version/err/"

mkdir -p "$bookkeeping_out_dir"
mkdir -p "$bookkeeping_err_dir"

cp -p -f /gscratch/stf/hardiksr/hiatus/logs/out/ar_paraphrase_jais_ifml1.txt "$bookkeeping_out_dir" 2>/dev/null
cp -p -f /gscratch/stf/hardiksr/hiatus/logs/err/ar_paraphrase_jais_ifml1.txt "$bookkeeping_err_dir" 2>/dev/null
