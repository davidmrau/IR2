#!/bin/bash
#SBATCH --job-name=exp_eval
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

#MODEL=~/IR2/results/exp_baseline_fixed/model_retrained_coverage_prior_0.05/model.gpu0.epoch0.step3106/
#BASELINE=~/IR2/results/exp_baseline_fixed/model_retrained_coverage/model.gpu0.epoch0.step3106/
MODEL=$1
BASELINE=2


echo 'prepare rouge model'
rm -f $MODEL/generated_model.txt
ls $MODEL/summary/ | sort -n | while read L; do 
	cat $MODEL/summary/$L | tr -d '\n\r' >> $MODEL/generated_model.txt
	echo >> $MODEL/generated_model.txt
done
echo 'done'

echo 'prepare target'
rm -f $MODEL/targets.txt
ls $MODEL/beam_ground_truth/ | sort -n | xargs -I {} sh -c "cat $MODEL/beam_ground_truth/{}" >> $MODEL/targets.txt
echo 'done'

echo 'prepare baseline rouge model'
rm -f $BASELINE/generated_baseline.txt
ls $BASELINE/summary/ | sort -n | while read L; do 
	cat $BASELINE/summary/$L | tr -d '\n\r' >> $BASELINE/generated_baseline.txt
	echo >> $BASELINE/generated_baseline.txt
done
echo 'done'

cd ~/vert/VertMetric/
echo 'calc wilcoxon'
python vert.py wilcoxon --generated=$MODEL/generated_model.txt --target=$MODEL/targets.txt --baseline=$BASELINE/generated_baseline.txt --out_dir=$MODEL > $MODEL/wilcoxon

cat $MODEL/wilcoxon

