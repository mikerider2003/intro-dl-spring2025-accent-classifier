#!/bin/bash
#SBATCH -p GPU              # partition (queue)
#SBATCH -N 1                # number of nodes
#SBATCH -t 0-36:00          # time (D-HH:MM)
#SBATCH -o cnn_1d_slurm.%N.%j.out  # STDOUT
#SBATCH -e cnn_1d_slurm.%N.%j.err  # STDERR
#SBATCH --gres=gpu:1        # request 1 GPU

# Setup Conda environment
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

# Activate your conda environment
source activate env1

# Navigate to your project directory
cd ~/intro-dl-spring2025-accent-classifier

# Run your training script
python train_1d_raw.py

