#!/bin/bash

#SBATCH --job-name=paraphrase_promptremix-tfs_ar3
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=150GB
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=80:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/paraphrase_promptremix-tfs_ar3.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/paraphrase_promptremix-tfs_ar3.err

source /mmfs1/home/hardiksr/.bashrc
conda activate hiatus
module load cuda gcc

prompts=$(ls prompts/prompts_ar/*.j2)
for prompt in $prompts
do
    echo "Processing $prompt"
    python scripts/2-paraphrase/paraphrase.py \
        --prompts $prompt \
        --model-name inceptionai/jais-adapted-70b-chat \
        --classifier-base-texts /gscratch/stf/hardiksr/hiatus/data/ar/raw/wiki_ar3.jsonl  \
        --output-dir /gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/wiki_ar3_paraphrases_jais
done

python scripts/2-paraphrase/process_paraphrased_wiki_data.py \
    --input-dir /gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/wiki_ar3_paraphrases_jais \
    --output-dir /gscratch/stf/hardiksr/hiatus/data/ar/paraphrases/wiki_ar3_paraphrases_jais_processed \
    --lang ar
