#!/bin/bash

#SBATCH --job-name=paraphrase-jais
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gpus=2
#SBATCH --cpus-per-task=6
#SBATCH --time=70:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/paraphrase_promptremix-ar2.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/paraphrase_promptremix-ar2.err

source /mmfs1/home/hardiksr/.bashrc
conda activate hiatus
module load cuda gcc

# prompts=$(ls prompts/prompts_ar/*.j2)
# for prompt in $prompts
# do
#     echo "Processing $prompt"
#     python scripts/2-paraphrase/paraphrase.py \
#         --prompts $prompt \
#         --model-name inceptionai/jais-adapted-70b-chat \
#         --classifier-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar2_classifiers.jsonl  \
#         --output-dir /gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/wiki_ar2_paraphrases_jais
# done

python scripts/2-paraphrase/process_paraphrased_wiki_data.py \
    --input-dir /gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/wiki_ar2_paraphrases_jais \
    --output-dir /gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/wiki_ar2_paraphrases_jais_processed \
    --lang ar
