#!/bin/bash
#SBATCH --account=def-htsang
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=32G
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --time=72:00:00
#SBATCH --job-name=rider_benchmark
#SBATCH --output=/home/anew/scratch/RIDER/benchmark_%j.out

export CUDA_VISIBLE_DEVICES=0

module load StdEnv/2023 apptainer/1.4.5
source ~/RIDER_env/bin/activate
cd ~/scratch/RIDER
python trainer_rl.py --config configs/default_rl.yaml
