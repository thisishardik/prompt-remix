#!/bin/bash

#SBATCH --job-name=train_classifier_promptremix-tfs_ar3
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=ALL
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=80:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/train_classifier_promptremix-tfs_ar3.out # where STDOUT goes
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/train_classifier_promptremix-tfs_ar3.err # where STDERR goes

source /mmfs1/home/hardiksr/.bashrc
conda activate hiatus
module load cuda gcc

styles=("descriptive,expository,narrative,persuasive" "formality" "function" "grade" "hostility" "length" "sarcasm" "voice" "vulgarity")

for style in "${styles[@]}"; do
    echo "Processing style $style"  
    python scripts/3-classifiers/train_classifier.py \
        --style "$style" \
        --model-name CAMeL-Lab/bert-base-arabic-camelbert-msa \
        --save-dir /gscratch/stf/hardiksr/hiatus/classifiers/ar3 \
        --data-dir /gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/wiki_ar3_paraphrases_jais_processed \
        --batch-size 128 \
        --max-seq-length 512 \
        --num-proc 6 \
        --num-train-epochs 15 \
        --experiment-suffix tfs_ar3
done


python scripts/3-classifiers/push_classifiers_to_hf.py \
    --model-name arabic-camelbert-msa-ar3  \
    --classifiers-dir /gscratch/stf/hardiksr/hiatus/classifiers/ar3 \