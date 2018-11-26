BASE=results/exp_tf_multihead_4_dropout
python2 main.py --tf_schedule --n_heads 4 --dropout_p_point --result_path $BASE >> $BASE/log.txt
