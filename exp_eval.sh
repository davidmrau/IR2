#!/bin/bash
#SBATCH --job-name=exp_eval
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

# example usage
#~/IR2$ sbatch exp_eval.sh results/exp_multihead/model_retrained_prior_loss_0_1_cov/ 'model.gpu0.epoch0.step17924' 4 rouge


module load cuDNN/7.0.5-CUDA-9.0.176
module purge
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH


MODELNAME=$2
OUTPUTDIR=~/IR2/$1/

TYPE_TEST=$4
if [ -z "$4" ]; then
    TYPE_TEST="rouge"
fi


mkdir -p $OUTPUTDIR/$MODELNAME

echo 'Check if summaries found'
if [ ! -d "$OUTPUTDIR/$MODELNAME/summary" ] || [ -z "$(ls $OUTPUTDIR/$MODELNAME/summary)" ]; then
    echo 'Not found, start generating summaries'
    python2 ~/IR2/main.py --n_heads $3 --result_path $OUTPUTDIR/$MODELNAME --predict --model_name $OUTPUTDIR/model/$MODELNAME --output_dir $OUTPUTDIR/$MODELNAME
    echo 'Done generating'
fi

echo 'Start ngramoverlap'
python2 ~/IR2/ngram_overlap.py $OUTPUTDIR/$MODELNAME ~/deepmind/test_set/test.pkl > $OUTPUTDIR/$MODELNAME/ngramoverlap
echo 'Done ngramoverlap'

rm -f $OUTPUTDIR/$MODELNAME/generated.txt
rm -f $OUTPUTDIR/$MODELNAME/targets.txt
echo 'Start prepare rouge'
ls $OUTPUTDIR/$MODELNAME/summary/ | sort -n | while read L; do 
	cat $OUTPUTDIR/$MODELNAME/summary/$L | tr -d '\n\r' >> $OUTPUTDIR/$MODELNAME/generated.txt
	echo >> $OUTPUTDIR/$MODELNAME/generated.txt
done
ls $OUTPUTDIR/$MODELNAME/beam_ground_truth/ | sort -n | xargs -I {} sh -c "cat $OUTPUTDIR/$MODELNAME/beam_ground_truth/{}" >> $OUTPUTDIR/$MODELNAME/targets.txt
echo 'Done prepare rouge'

cd ~/vert/VertMetric/
echo 'Start rouge'
python vert.py $TYPE_TEST --generated=$OUTPUTDIR/$MODELNAME/generated.txt --target=$OUTPUTDIR/$MODELNAME/targets.txt --out_dir=$OUTPUTDIR/$MODELNAME --rouge_type f-measure > $OUTPUTDIR/$MODELNAME/verts
echo 'Done rouge'

cat $OUTPUTDIR/$MODELNAME/verts $OUTPUTDIR/$MODELNAME/ngramoverlap
