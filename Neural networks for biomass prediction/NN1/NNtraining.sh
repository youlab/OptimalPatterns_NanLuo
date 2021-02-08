#!/bin/bash
#SBATCH --job-name=training_NN
#SBATCH --output=logs/array_%A.out
#SBATCH -p scavenger-gpu --gres=gpu:1
#SBATCH --mem=10G 
#SBATCH -c 6
 
module load Python-GPU/3.6.5
python NN_predict_bbiomass.py

# happy end
exit 0
