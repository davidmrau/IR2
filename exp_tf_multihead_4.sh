BASE=results/exp_tf_multihead_4
python2 main.py --tf_schedule --n_heads 4 --result_path $BASE >> $BASE/log.txt
