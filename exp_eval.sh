#!/bin/bash
#SBATCH --job-name=exp_eval
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

# example usage
#~/IR2$ sbatch exp_eval.sh results/exp_multihead/model_retrained_prior_loss_0_1_cov/ 'model.gpu0.epoch0.step17924' 4





module load cuDNN/7.0.5-CUDA-9.0.176
module purge
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH


MODELNAME=$2
OUTPUTDIR=~/IR2/$1/

#MODELFOLDER=~/IR2/$BASE/$4/


mkdir -p $OUTPUTDIR/$MODELNAME

echo $OUTPUTDIR

python2 ~/IR2/main.py --n_heads $3 --result_path $OUTPUTDIR/$MODELNAME --predict --model_name $OUTPUTDIR/model/$MODELNAME --output_dir $OUTPUTDIR/$MODELNAME
python2 ~/IR2/prepare_rouge.py --result_path $OUTPUTDIR/$MODELNAME


python2 ~/IR2/ngram_overlap.py $OUTPUTDIR/$MODELNAME ~/deepmind/test_set/test.pkl > $OUTPUTDIR/$MODELNAME/ngramoverlap
#python2 ~/IR2/ngram_overlap.py $OUTPUTDIR ~/deepmind/test_set/test_1100.pkl > $OUTPUTDIR/ngramoverlap
cat $OUTPUTDIR/$MODELNAME/ngramoverlap
rm $OUTPUTDIR/$MODELNAME/generated.txt
rm $OUTPUTDIR/$MODELNAME/targets.txt
ls $OUTPUTDIR/$MODELNAME/summary/ | sort -n | while read L; do 
	cat $OUTPUTDIR/$MODELNAME/summary/$L | tr -d '\n\r' >> $OUTPUTDIR/$MODELNAME/generated.txt
	echo >> $OUTPUTDIR/$MODELNAME/generated.txt
done
ls $OUTPUTDIR/$MODELNAME/beam_ground_truth/ | sort -n | xargs -I {} sh -c "cat $OUTPUTDIR/$MODELNAME/beam_ground_truth/{}" >> $OUTPUTDIR/$MODELNAME/targets.txt

cd ~/VertMetric/
python vert.py score \
	 --generated=$OUTPUTDIR/$MODELNAME/generated.txt \
	  --target=$OUTPUTDIR/$MODELNAME/targets.txt \
	   --out_dir=$OUTPUTDIR/$MODELNAME \
	    --rouge_type f-measure
