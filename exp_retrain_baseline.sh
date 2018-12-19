#!/bin/bash
#SBATCH --job-name=baseline_retrain
#SBATCH --ntasks=1
#SBATCH --time=1:30:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

module load cuDNN/7.0.5-CUDA-9.0.176
module purge
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

BASE=/home/lgpu0235/IR2/results/exp_baseline_fixed
RESULTPATH=$BASE"/model_retrained_coverage_prior_$1/"

mkdir -p $RESULTPATH
python2 ~/IR2/main.py --n_heads 1 --result_path $RESULTPATH --retrain --model_name $BASE"/model/model.gpu0.epoch12.step229514" --use_w_prior_point_loss --w_prior_point_scalar $1
