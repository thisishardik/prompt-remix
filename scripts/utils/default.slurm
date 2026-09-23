#!/bin/bash

#SBATCH --job-name={model}_{job}_{task}
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
#SBATCH --output=/gscratch/stf/mpotto/hiatus/logs/{model}_{task}_{account}_{partition}.out
#SBATCH --error=/gscratch/stf/mpotto/hiatus/logs/{model}_{task}_{account}_{partition}.err

source /mmfs1/home/mpotto/.bashrc
cd /mmfs1/home/mpotto/code/hiatus/scripts
conda activate hiatus
module load cuda gcc
