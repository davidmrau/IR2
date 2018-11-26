BASE=results/exp_tf_multihead_4_w_prior_point_loss
python2 main.py --tf_schedule --n_heads 4 --use_w_prior_point_loss --result_path $BASE >> $BASE/log.txt
