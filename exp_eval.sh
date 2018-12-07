#!/bin/bash
#SBATCH --job-name=exp_eval
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

module load cuDNN/7.0.5-CUDA-9.0.176
module purge
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH


BASE=$1
MODEL=$2

OUTPUTDIR=$BASE/$MODEL/
mkdir -p $OUTPUTDIR

python2 ~/IR2/main.py --n_heads $3 --result_path $BASE --predict --model_name $MODEL --output_dir $OUTPUTDIR
python2 ~/IR2/prepare_rouge.py --result_path $OUTPUTDIR
cd ~/Home-of-ROUGE-1.5.5/
perl ROUGE-1.5.5.pl $OUTPUTDIR/myROUGE_Config.xml C > $OUTPUTDIR/rouge

cat $OUTPUTDIR/rouge

python2 ~/IR2/ngram_overlap.py $OUTPUTDIR > $OUTPUTDIR/ngramoverlap

cat $OUTPUTDIR/ngramoverlap
