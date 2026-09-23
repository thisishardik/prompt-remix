#!/bin/bash

#SBATCH --job-name=classifiers_promptremix-zh0
#SBATCH --mail-user=mpotto@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --time=80:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/mpotto/code/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/mpotto/hiatus/logs/classifiers_promptremix-zh0.out
#SBATCH --error=/gscratch/stf/mpotto/hiatus/logs/classifiers_promptremix-zh0.err

source /mmfs1/home/mpotto/.bashrc
conda activate hiatus
module load cuda gcc

styles=("descriptive,expository,narrative,persuasive" "formality" "function" "grade" "hostility" "length" "sarcasm" "voice" "vulgarity")
for style in "${styles[@]}"; do
    echo "Processing style $style"  
    python scripts/classifiers/train_classifier.py \
        --model-name hfl/chinese-roberta-wwm-ext-large \
        --data-dir /gscratch/stf/mpotto/hiatus/data/zh/paraphrases/wiki_zh0_paraphrases_qwen_processed \
        --save-dir /gscratch/stf/mpotto/hiatus/classifiers/zh \
        --max-length 512 \
        --style $style \
        --num-proc 6 \
        --num-train-epochs 10
done

python scripts/3-classifiers/push_classifiers_to_hf.py \
    --model-name hfl/chinese-roberta-wwm-ext-large  \
    --classifiers-dir /gscratch/stf/mpotto/hiatus/classifiers/zh \
