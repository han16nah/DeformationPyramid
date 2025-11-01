#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A100:1,gpumem_per_gpu:80GB
#SBATCH --time=72:00:00
#SBATCH --mem=265gb

module load devel/cuda/11.6
module load devel/miniforge/24.9.2
mamba activate ndp
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64' # ,garbage_collection_threshold:0.6'
# export CUDA_VISIBLE_DEVICES=1,2,3,4
cd correspondence
python main.py configs/train/plants_S5_sub.yaml

