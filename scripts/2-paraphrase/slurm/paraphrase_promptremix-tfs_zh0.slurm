#!/bin/bash

#SBATCH --job-name=paraphrase_promptremix-zh0
#SBATCH --mail-user=mpotto@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=80:00:00

#SBATCH --chdir=/mmfs1/home/mpotto/code/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/mpotto/hiatus/logs/paraphrase_promptremix-zh0.out
#SBATCH --error=/gscratch/stf/mpotto/hiatus/logs/paraphrase_promptremix-zh0.err


source /mmfs1/home/mpotto/.bashrc
conda activate hiatus
module load cuda gcc

prompts=$(ls prompts/prompts_zh/*.j2)
for prompt in $prompts
do
    echo "Processing $prompt"
    python scripts/2-paraphrase/paraphrase.py \
        --prompts $prompt \
        --model-name Qwen/QwQ-32B \
        --adapter-base-texts /gscratch/stf/mpotto/hiatus/data/zh/raw/wiki_zh0_adapters.jsonl \
        --classifier-base-texts /gscratch/stf/mpotto/hiatus/data/zh/raw/wiki_zh0_classifiers.jsonl  \
        --output-dir /gscratch/stf/mpotto/hiatus/data/zh/paraphrases/wiki_zh0_paraphrases_qwen
done

python scripts/2-paraphrases/process_paraphrased_wiki_data.py \
    --input-dir /gscratch/stf/mpotto/hiatus/data/zh/paraphrases/wiki_zh0_paraphrases_qwen \
    --output-dir /gscratch/stf/mpotto/hiatus/data/zh/paraphrases/wiki_zh0_paraphrases_qwen_processed \
