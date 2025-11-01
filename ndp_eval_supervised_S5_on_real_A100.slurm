#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100:1,gpumem_per_gpu:80GB
#SBATCH --time=10:00:00
#SBATCH --mem=64gb

module load devel/cuda/11.6
module load devel/miniforge/24.9.2
mamba activate ndp
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python eval_supervised_noreport.py --config config/LNDP_S5_on_real.yaml --write
