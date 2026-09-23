#!/bin/bash
#SBATCH --job-name=pr_random_summary
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/pr_random_summary.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/pr_random_summary.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm

python ablations/random-axis/summarize_random_axis.py
