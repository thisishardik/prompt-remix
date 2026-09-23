#!/bin/bash

#SBATCH --job-name=download_authbench
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/download_authbench.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/download_authbench.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm

python scripts/7-baselines/authbench/download_authbench.py \
    --output-dir /gscratch/ifml1/hiatus/data/raw/authbench \
    --languages en ru zh ar
