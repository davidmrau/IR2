BASE=results/exp_tf_multihead_4_dropout
mkdir -p $BASE
python2 main.py --tf_schedule --tf_offset 200000 --n_heads 4 --dropout_p_point --result_path $BASE >> $BASE/log.txt
