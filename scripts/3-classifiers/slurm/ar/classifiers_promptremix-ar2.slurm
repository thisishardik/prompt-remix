#!/bin/bash

#SBATCH --job-name=classifiers_promptremix-ar2
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=150GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=50:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/classifiers_promptremix-ar2.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/classifiers_promptremix-ar2.err

source /mmfs1/home/hardiksr/.bashrc
conda activate hiatus
module load cuda gcc

styles=("descriptive,expository,narrative,persuasive" "formality" "function" "grade" "hostility" "length" "sarcasm" "voice" "vulgarity")
# for style in "${styles[@]}"; do
#     echo "Processing style $style"  
#     python scripts/3-classifiers/train_classifier.py \
#         --model-name CAMeL-Lab/bert-base-arabic-camelbert-msa \
#         --data-dir /gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/wiki_ar2_paraphrases_jais_processed \
#         --save-dir /gscratch/stf/hardiksr/hiatus/models/classifiers/ar2 \
#         --max-seq-length 512 \
#         --batch-size 128 \
#         --style $style \
#         --num-proc 5 \
#         --num-train-epochs 200
# done

python scripts/3-classifiers/push_classifiers_to_hf.py \
    --model-name arabic-camelbert-msa-ar2  \
    --classifiers-dir /gscratch/stf/hardiksr/hiatus/classifiers/ar2 \
