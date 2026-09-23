#!/bin/bash

#SBATCH --job-name=train_classifier
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=ALL
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=80:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus-dev
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/out/ar_train_classifier_ifml1.txt # where STDOUT goes
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/err/ar_train_classifier_ifml1.txt # where STDERR goes

source /mmfs1/home/hardiksr/.bashrc
conda activate hiatus
module load cuda gcc

data_dir="/gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/$version/ar_paraphrases_jais/"
styles=("descriptive,expository,narrative,persuasive" "formality" "function" "grade" "hostility" "length" "sarcasm" "voice" "vulgarity")

for style in "${styles[@]}"; do
    echo "Processing style $style"  
    python scripts/3-classifiers/train_classifier.py \
        --style "$style" \
        --model_name CAMeL-Lab/bert-base-arabic-camelbert-msa \
        --save_dir /gscratch/stf/hardiksr/hiatus/models/ \
        --data_dir $data_dir \
        --batch_size 128 \
        --num_proc 6 \
        --num_train_epochs 20
done


# Bookkeeping to track metrics

bookkeeping_out_dir="/gscratch/stf/hardiksr/hiatus/bookkeeping/$version/out/"
bookkeeping_err_dir="/gscratch/stf/hardiksr/hiatus/bookkeeping/$version/err/"

mkdir -p "$bookkeeping_out_dir"
mkdir -p "$bookkeeping_err_dir"

cp -p -f /gscratch/stf/hardiksr/hiatus/logs/out/ar_train_classifier_ifml1.txt "$bookkeeping_out_dir" 2>/dev/null
cp -p -f /gscratch/stf/hardiksr/hiatus/logs/err/ar_train_classifier_ifml1.txt "$bookkeeping_err_dir" 2>/dev/null