#!/bin/bash

#SBATCH --job-name=classifiers_promptremix-ar1
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --time=80:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/classifiers_promptremix-ar1.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/classifiers_promptremix-ar1.err

source /mmfs1/home/hardiksr/.bashrc
cd /mmfs1/home/hardiksr/repository/hiatus/scripts
conda activate hiatus
module load cuda gcc

styles=("descriptive,expository,narrative,persuasive" "formality" "function" "grade" "hostility" "length" "sarcasm" "voice" "vulgarity")
for style in "${styles[@]}"; do
    echo "Processing style $style"  
    python scripts/classifiers/train_classifier.py \
        --model-name CAMeL-Lab/bert-base-arabic-camelbert-msa \
        --data-dir /gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/wiki_ar1_paraphrases_jais_processed \
        --save-dir /gscratch/stf/hardiksr/hiatus/classifiers/ar \
        --max-length 512 \
        --batch-size 128 \
        --style $style \
        --num-proc 6 \
        --num-train-epochs 100
done

python scripts/3-classifiers/push_classifiers_to_hf.py \
    --model-name hfl/arabic-bert-camelbert-msa  \
    --classifiers-dir /gscratch/stf/hardiksr/hiatus/classifiers/ar1 \
