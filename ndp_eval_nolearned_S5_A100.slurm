#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100:1
#SBATCH --time=24:00:00
#SBATCH --mem=32gb

module load devel/cuda/11.6
module load devel/miniforge/24.9.2
mamba activate ndp
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python eval_nolearned_adapted.py --config config/NDP_S5.yaml --write
