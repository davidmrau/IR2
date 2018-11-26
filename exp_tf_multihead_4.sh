BASE=results/exp_tf_multihead_4
mkdir -p $BASE
python2 main.py --tf_schedule --tf_offset 200000 --n_heads 4 --result_path $BASE >> $BASE/log.txt
