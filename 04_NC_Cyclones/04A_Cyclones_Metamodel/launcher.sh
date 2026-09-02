#!/bin/bash
#SBATCH --job-name=kf_gauss  # Job name
#SBATCH --partition=geocean_priority      # Standard output and error log
#SBATCH --mem=64G                 # Memory per node in GB (see also --mem-per-cpu)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=26       # Number of tasks per node

source /nfs/home/geocean/faugeree/miniforge3/etc/profile.d/conda.sh
conda activate work

python -u k_fold_gaussian_final.py