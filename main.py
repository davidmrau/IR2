# -*- coding: utf-8 -*-
import os
cudaid = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid)

import sys
import time
import numpy as np
import cPickle as pickle
import copy
import random
from random import shuffle
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import data as datar
from model import *
from utils_pg import *
from configs import *
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--colab', help='Flag whether running on colab', action='store_true')

parser.add_argument('--debug', help='Flag whether running in debug mode', action='store_true')

parser.add_argument('--predict', help='Flag whether to predict or to train', action='store_true')
parser.add_argument('--tf_schedule', help='using Teacher forcing schedule', action='store_true')
parser.add_argument('--batch_size', help='Batch size for training', default=16, type=int)
parser.add_argument('--tf_offset', help='offset for teacher forcing scheduler', default=350000, type=int)

parser.add_argument('--dropout_p_point', help='Chance of dropping out p_point', default=0.0, type=float)


parser.add_argument('--p_point_scalar', help='scalar for p_point loss', default=1.0, type=float)
parser.add_argument('--use_p_point_loss', help='use p_point to the loss ', action='store_true')

parser.add_argument('--use_w_prior_point_loss', help='use w prior point loss ', action='store_true')
parser.add_argument('--retrain', help='set flag whether retraining ', action='store_true')
parser.add_argument('--w_prior_point_scalar', help='scalar for w prior point loss', default=1.0, type=float)


parser.add_argument('--result_path', help='path where the model and results will be stored', default='result', type=str)
parser.add_argument('--model_name', help='model file name that should be continued training', default='', type=str)
parser.add_argument('--n_heads', help='number of attention heads', default=4, type=int)
parser.add_argument('--output_dir', help='Where to save the summaries/ground truth', default='', type=str)
parser.add_argument('--model_folder', help='Where to look for models', default='', type=str)
opt = parser.parse_args()

cfg = DeepmindConfigs(opt.colab,opt.result_path,opt.n_heads)

cfg.cc.BEAM_SUMM_PATH = opt.output_dir + "/beam_summary/"
cfg.cc.BEAM_GT_PATH = opt.output_dir + "/beam_ground_truth/"
cfg.cc.GROUND_TRUTH_PATH = opt.output_dir + "/ground_truth/"
cfg.cc.SUMM_PATH = opt.output_dir + "/summary/"
cfg.cc.TMP_PATH = opt.output_dir  + "/tmp/"

TRAINING_DATASET_CLS = DeepmindTraining(opt.batch_size)
TESTING_DATASET_CLS = DeepmindTesting


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_folder(opt.result_path)
create_folder(opt.result_path+'/tmp')
create_folder(opt.result_path+'/model')
create_folder(opt.result_path+'/beam_summary')
create_folder(opt.result_path+'/beam_ground_truth')
create_folder(opt.result_path+'/ground_truth')
create_folder(opt.result_path+'/summary')



def print_basic_info(modules, consts, options):
    if opt.debug:
        print "\nWARNING: IN DEBUGGING MODE\n"
    if options["copy"]:
        print "USE COPY MECHANISM"
    if options["coverage"]:
        print "USE COVERAGE MECHANISM"
    if  options["avg_nll"]:
        print "USE AVG NLL as LOSS"
    else:
        print "USE NLL as LOSS"
    if options["has_learnable_w2v"]:
        print "USE LEARNABLE W2V EMBEDDING"
    if options["is_bidirectional"]:
        print "USE BI-DIRECTIONAL RNN"

    if options["use_p_point_loss"]:
        print "USE P_POINT LOSS"


    if options["omit_eos"]:
        print "<eos> IS OMITTED IN TESTING DATA"
    if options["prediction_bytes_limitation"]:
        print "MAXIMUM BYTES IN PREDICTION IS LIMITED"
    print "RNN TYPE: " + options["cell"]
    for k in consts:
        print k + ":", consts[k]

def init_modules():

    init_seeds()

    options = {}
    options["is_predicting"] = opt.predict
    options["use_p_point_loss"] = opt.use_p_point_loss
    options["use_w_prior_point_loss"] = opt.use_w_prior_point_loss
    options["tf_offset_decay"] = opt.tf_offset

    options["cuda"] = cfg.CUDA and torch.cuda.is_available()
    options["device"] = torch.device("cuda" if  options["cuda"] else "cpu")
    print('Running', options["device"])
    #in config.py
    options["cell"] = cfg.CELL
    options["copy"] = cfg.COPY
    options["coverage"] = cfg.COVERAGE
    options["is_bidirectional"] = cfg.BI_RNN
    options["avg_nll"] = cfg.AVG_NLL
    options['retrain']  = opt.retrain
    options["beam_decoding"] = cfg.BEAM_SEARCH # False for greedy decoding

    assert TRAINING_DATASET_CLS.IS_UNICODE == TESTING_DATASET_CLS.IS_UNICODE
    options["is_unicode"] = TRAINING_DATASET_CLS.IS_UNICODE # True Chinese dataet
    options["has_y"] = TRAINING_DATASET_CLS.HAS_Y
    options["is_debugging"] = opt.debug
    options["has_learnable_w2v"] = True
    options["omit_eos"] = False # omit <eos> and continuously decode until length of sentence reaches MAX_LEN_PREDICT (for DUC testing data)
    options["prediction_bytes_limitation"] = False if TESTING_DATASET_CLS.MAX_BYTE_PREDICT == None else True

    assert options["is_unicode"] == False

    consts = {}

    consts["idx_gpu"] = cudaid

    consts["p_point_scalar"] = opt.p_point_scalar
    consts["w_prior_point_scalar"] = opt.w_prior_point_scalar
    consts["norm_clip"] = cfg.NORM_CLIP
    consts["dim_x"] = cfg.DIM_X
    consts["dim_y"] = cfg.DIM_Y
    consts["len_x"] = cfg.MAX_LEN_X + 1 # plus 1 for eos
    consts["len_y"] = cfg.MAX_LEN_Y + 1
    consts["num_x"] = cfg.MAX_NUM_X
    consts["num_y"] = cfg.NUM_Y
    consts["hidden_size"] = cfg.HIDDEN_SIZE

    consts["n_heads"] = cfg.N_HEADS
    consts["dropout_p_point"] = opt.dropout_p_point

    consts["batch_size"] = 5 if opt.debug else TRAINING_DATASET_CLS.BATCH_SIZE
    if opt.debug:
        consts["testing_batch_size"] = 1 if options["beam_decoding"] else 2
    else:
        #consts["testing_batch_size"] = 1 if options["beam_decoding"] else TESTING_DATASET_CLS.BATCH_SIZE
        consts["testing_batch_size"] = TESTING_DATASET_CLS.BATCH_SIZE

    consts["min_len_predict"] = TESTING_DATASET_CLS.MIN_LEN_PREDICT
    consts["max_len_predict"] = TESTING_DATASET_CLS.MAX_LEN_PREDICT
    consts["max_byte_predict"] = TESTING_DATASET_CLS.MAX_BYTE_PREDICT
    consts["testing_print_size"] = TESTING_DATASET_CLS.PRINT_SIZE

    consts["lr"] = cfg.LR
    consts["beam_size"] = cfg.BEAM_SIZE

    consts["max_epoch"] = 14 if opt.debug else 13
    consts["print_time"] = 150 if opt.retrain else 70
    consts["save_epoch"] = 1

    assert consts["dim_x"] == consts["dim_y"]
    assert consts["beam_size"] >= 1

    modules = {}

    [_, dic, hfw, w2i, i2w, w2w] = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "dic.pkl", "r"))
    modules["priors"] = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "prior.pkl", "rb"))
    consts["dict_size"] = len(dic)
    modules["dic"] = dic
    modules["w2i"] = w2i
    modules["i2w"] = i2w
    modules["lfw_emb"] = modules["w2i"][cfg.W_UNK]
    modules["eos_emb"] = modules["w2i"][cfg.W_EOS]
    consts["pad_token_idx"] = modules["w2i"][cfg.W_PAD]
    
    if opt.model_folder:
        cfg.cc.MODEL_PATH = opt.model_folder
    return modules, consts, options
def teacher_forcing_ratio(steps, offset):
    if steps < offset:
        return False
    else:
        prob_tf =  100/(100 + np.exp(steps-offset/step_size))
        return random.random() < prob_tf


def greedy_decode(flist, batch, model, modules, consts, options):
    testing_batch_size = len(flist)

    dec_result = [[] for i in xrange(testing_batch_size)]
    existence = [True] * testing_batch_size
    num_left = testing_batch_size

    if options["copy"]:
        x, word_emb, dec_state, x_mask, y, len_y, ref_sents, max_ext_len, oovs = batch
    else:
        x, word_emb, dec_state, x_mask, y, len_y, ref_sents = batch

    next_y = torch.LongTensor(-np.ones((1, testing_batch_size), dtype="int64")).to(options["device"])

    if options["cell"] == "lstm":
        dec_state = (dec_state, dec_state)
    if options["coverage"]:
        acc_att = Variable(torch.zeros(T.transpose(x, 0, 1).size())).to(options["device"]) # B *len(x)

    for step in xrange(consts["max_len_predict"]):
        if num_left == 0:
            break
        if options["copy"] and options["coverage"]:
            y_pred, dec_state, acc_att = model.decode_once(next_y, word_emb, dec_state, x_mask, x, max_ext_len, acc_att)
        elif options["copy"]:
            y_pred, dec_state = model.decode_once(next_y, word_emb, dec_state, x_mask, x, max_ext_len)
        elif options["coverage"]:
            y_pred, dec_state, acc_att = model.decode_once(next_y, word_emb, dec_state, x_mask, acc_att=acc_att)
        else:
            y_pred, dec_state = model.decode_once(next_y, word_emb, dec_state, x_mask)

        dict_size = y_pred.shape[-1]
        y_pred = y_pred.view(testing_batch_size, dict_size)
        next_y_ = torch.argmax(y_pred, 1)
        next_y = []
        for e in range(testing_batch_size):
            eid = next_y_[e].item()
            if eid in modules["i2w"]:
                next_y.append(eid)
            else:
                next_y.append(modules["lfw_emb"]) # unk for copy mechanism
        next_y = np.array(next_y).reshape((1, testing_batch_size))
        next_y = torch.LongTensor(next_y).to(options["device"])

        if options["coverage"]:
            acc_att = acc_att.view(testing_batch_size, acc_att.shape[-1])

        if options["cell"] == "lstm":
            dec_state = (dec_state[0].view(testing_batch_size, dec_state[0].shape[-1]), dec_state[1].view(testing_batch_size, dec_state[1].shape[-1]))
        else:
            dec_state = dec_state.view(testing_batch_size, dec_state.shape[-1])

        for idx_doc in xrange(testing_batch_size):
            if existence[idx_doc] == False:
                continue

            idx_max = next_y[0, idx_doc].item()
            if idx_max == modules["eos_emb"] and len(dec_result[idx_doc]) >= consts["min_len_predict"]:
                existence[idx_doc] = False
                num_left -= 1
            else:
                dec_result[idx_doc].append(str(idx_max))

    # for task with bytes-length limitation
    if options["prediction_bytes_limitation"]:
        for i in xrange(len(dec_result)):
            sample = dec_result[i]
            b = 0
            for j in xrange(len(sample)):
                e = int(sample[j])
                if e in modules["i2w"]:
                    word = modules["i2w"][e]
                else:
                    word = oovs[e - len(modules["i2w"])]
                if j == 0:
                    b += len(word)
                else:
                    b += len(word) + 1
                if b > consts["max_byte_predict"]:
                    sorted_samples[i] = sorted_samples[i][0 : j]
                    break

    for idx_doc in xrange(testing_batch_size):
        fname = str(flist[idx_doc])
        if len(dec_result[idx_doc]) >= consts["min_len_predict"]:
            dec_words = []
            for e in dec_result[idx_doc]:
                e = int(e)
                if e in modules["i2w"]: # if not copy, the word are all in dict
                    dec_words.append(modules["i2w"][e])
                else:
                    dec_words.append(oovs[e - len(modules["i2w"])])
            write_for_rouge(fname, ref_sents[idx_doc], dec_words, cfg)
        else:
            print "ERROR: " + fname


def beam_decode(fname, batch, model, modules, consts, options):
    fname = str(fname)

    beam_size = consts["beam_size"]
    num_live = 1
    num_dead = 0
    samples = []
    sample_scores = np.zeros(beam_size)

    last_traces = [[]]
    last_scores = torch.FloatTensor(np.zeros(1)).to(options["device"])
    last_states = []

    if options["copy"]:
        x, word_emb, dec_state, x_mask, y, len_y, ref_sents, max_ext_len, oovs = batch
    else:
        x, word_emb, dec_state, x_mask, y, len_y, ref_sents = batch

    next_y = torch.LongTensor(-np.ones((1, num_live), dtype="int64")).to(options["device"])
    x = x.unsqueeze(1)
    word_emb = word_emb.unsqueeze(1)
    x_mask = x_mask.unsqueeze(1)
    dec_state = dec_state.unsqueeze(0)
    if options["cell"] == "lstm":
        dec_state = (dec_state, dec_state)

    if options["coverage"]:
        acc_att = Variable(torch.zeros(T.transpose(x, 0, 1).size())).to(options["device"]) # B *len(x)
        last_acc_att = []

    for step in xrange(consts["max_len_predict"]):
        tile_word_emb = word_emb.repeat(1, num_live, 1)
        tile_x_mask = x_mask.repeat(1, num_live, 1)
        if options["copy"]:
            tile_x = x.repeat(1, num_live)

        if options["copy"] and options["coverage"]:
            y_pred, dec_state, acc_att = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask, x=tile_x, max_ext_len=max_ext_len, acc_att=acc_att)
        elif options["copy"]:
            y_pred, dec_state = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask, x=tile_x, max_ext_len=max_ext_len)
        elif options["coverage"]:
            y_pred, dec_state, acc_att = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask, acc_att=acc_att)
        else:
            y_pred, dec_state = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask)
        dict_size = y_pred.shape[-1]
        y_pred = y_pred.view(num_live, dict_size)
        if options["coverage"]:
            acc_att = acc_att.view(num_live, acc_att.shape[-1])

        if options["cell"] == "lstm":
            dec_state = (dec_state[0].view(num_live, dec_state[0].shape[-1]), dec_state[1].view(num_live, dec_state[1].shape[-1]))
        else:
            dec_state = dec_state.view(num_live, dec_state.shape[-1])

        cand_scores = last_scores + torch.log(y_pred) # larger is better
        cand_scores = cand_scores.flatten()
        idx_top_joint_scores = torch.topk(cand_scores, beam_size - num_dead)[1]


        idx_last_traces = idx_top_joint_scores / dict_size
        idx_word_now = idx_top_joint_scores % dict_size
        top_joint_scores = cand_scores[idx_top_joint_scores]

        traces_now = []
        scores_now = np.zeros((beam_size - num_dead))
        states_now = []
        if options["coverage"]:
            acc_att_now = []
            last_acc_att = []

        for i, [j, k] in enumerate(zip(idx_last_traces, idx_word_now)):
            traces_now.append(last_traces[j] + [k])
            scores_now[i] = copy.copy(top_joint_scores[i])
            if options["cell"] == "lstm":
                states_now.append((copy.copy(dec_state[0][j, :]), copy.copy(dec_state[1][j, :])))
            else:
                states_now.append(copy.copy(dec_state[j, :]))
            if options["coverage"]:
                acc_att_now.append(copy.copy(acc_att[j, :]))

        num_live = 0
        last_traces = []
        last_scores = []
        last_states = []
        for i in xrange(len(traces_now)):
            if traces_now[i][-1] == modules["eos_emb"] and len(traces_now[i]) >= consts["min_len_predict"]:
                samples.append([str(e.item()) for e in traces_now[i][:-1]])
                sample_scores[num_dead] = scores_now[i]
                num_dead += 1
            else:
                last_traces.append(traces_now[i])
                last_scores.append(scores_now[i])
                last_states.append(states_now[i])
                if options["coverage"]:
                    last_acc_att.append(acc_att_now[i])
                num_live += 1
        if num_live == 0 or num_dead >= beam_size:
            break

        last_scores = torch.FloatTensor(np.array(last_scores).reshape((num_live, 1))).to(options["device"])
        next_y = []
        for e in last_traces:
            eid = e[-1].item()
            if eid in modules["i2w"]:
                next_y.append(eid)
            else:
                next_y.append(modules["lfw_emb"]) # unk for copy mechanism

        next_y = np.array(next_y).reshape((1, num_live))
        next_y = torch.LongTensor(next_y).to(options["device"])
        if options["cell"] == "lstm":
            h_states = []
            c_states = []
            for state in last_states:
                h_states.append(state[0])
                c_states.append(state[1])
            dec_state = (torch.stack(h_states).view((num_live, h_states[0].shape[-1])),\
                         torch.stack(c_states).view((num_live, c_states[0].shape[-1])))
        else:
            dec_state = torch.stack(last_states).view((num_live, dec_state.shape[-1]))
        if options["coverage"]:
            acc_att = torch.stack(last_acc_att).view((num_live, acc_att.shape[-1]))

        assert num_live + num_dead == beam_size

    if num_live > 0:
        for i in xrange(num_live):
            samples.append([str(e.item()) for e in last_traces[i]])
            sample_scores[num_dead] = last_scores[i]
            num_dead += 1

    #weight by length
    for i in xrange(len(sample_scores)):
        sent_len = float(len(samples[i]))
        sample_scores[i] = sample_scores[i] / sent_len #avg is better than sum.   #*  math.exp(-sent_len / 10)

    idx_sorted_scores = np.argsort(sample_scores) # ascending order
    if options["has_y"]:
        ly = len_y[0]
        y_true = y[0 : ly].tolist()
        y_true = [str(i) for i in y_true[:-1]] # delete <eos>

    sorted_samples = []
    sorted_scores = []
    filter_idx = []
    for e in idx_sorted_scores:
        if len(samples[e]) >= consts["min_len_predict"]:
            filter_idx.append(e)
    if len(filter_idx) == 0:
        filter_idx = idx_sorted_scores
    for e in filter_idx:
        sorted_samples.append(samples[e])
        sorted_scores.append(sample_scores[e])

    num_samples = len(sorted_samples)
    if len(sorted_samples) == 1:
        sorted_samples = sorted_samples[0]
        num_samples = 1

    # for task with bytes-length limitation
    if options["prediction_bytes_limitation"]:
        for i in xrange(len(sorted_samples)):
            sample = sorted_samples[i]
            b = 0
            for j in xrange(len(sample)):
                e = int(sample[j])
                if e in modules["i2w"]:
                    word = modules["i2w"][e]
                else:
                    word = oovs[e - len(modules["i2w"])]
                if j == 0:
                    b += len(word)
                else:
                    b += len(word) + 1
                if b > consts["max_byte_predict"]:
                    sorted_samples[i] = sorted_samples[i][0 : j]
                    break

    dec_words = []

    for e in sorted_samples[-1]:
        e = int(e)
        if e in modules["i2w"]: # if not copy, the word are all in dict
            dec_words.append(modules["i2w"][e])
        else:
            dec_words.append(oovs[e - len(modules["i2w"])])

    write_for_rouge(fname, ref_sents, dec_words, cfg)

    # beam search history for checking
    if not options["copy"]:
        oovs = None
    write_summ("".join((cfg.cc.BEAM_SUMM_PATH, fname)), sorted_samples, num_samples, options, modules["i2w"], oovs, sorted_scores)
    write_summ("".join((cfg.cc.BEAM_GT_PATH, fname)), y_true, 1, options, modules["i2w"], oovs)


def predict(model, modules, consts, options):
    print "start predicting,"
    options["has_y"] = TESTING_DATASET_CLS.HAS_Y
    if options["beam_decoding"]:
        print "using beam search"
    else:
        print "using greedy search"
    rebuild_dir(cfg.cc.BEAM_SUMM_PATH)
    rebuild_dir(cfg.cc.BEAM_GT_PATH)
    rebuild_dir(cfg.cc.GROUND_TRUTH_PATH)
    rebuild_dir(cfg.cc.SUMM_PATH)

    print "loading test set..."
    if opt.debug:
        xy_list = pickle.load(open(cfg.cc.TESTING_DATA_PATH + "test_500.pkl", "r"))
    else:
        xy_list = pickle.load(open(cfg.cc.TESTING_DATA_PATH + "test.pkl", "r"))
    batch_list, num_files, num_batches = datar.batched(len(xy_list), options, consts)
    
    # Save order of batches for ngram overlap
    batches_sorted_idx = []

    print "num_files = ", num_files, ", num_batches = ", num_batches

    running_start = time.time()
    partial_num = 0
    total_num = 0
    si = 0
    for idx_batch in xrange(num_batches):
        test_idx = batch_list[idx_batch]
        batch_raw = [xy_list[xy_idx] for xy_idx in test_idx]
        batch = datar.get_data(batch_raw, modules, consts, options)

        assert len(test_idx) == batch.x.shape[1] # local_batch_size

        x, len_x, x_mask, y, len_y, y_mask, oy, x_ext, y_ext, oovs, batch_sorted_idx = sort_samples(batch.x, batch.len_x, \
                                                             batch.x_mask, batch.y, batch.len_y, batch.y_mask, \
                                                             batch.original_summarys, batch.x_ext, batch.y_ext, batch.x_ext_words, 
                                                             return_idx=True)
        batches_sorted_idx.append(batch_sorted_idx)
        

        word_emb, dec_state = model.encode(torch.LongTensor(x).to(options["device"]),\
                                           torch.LongTensor(len_x).to(options["device"]),\
                                           torch.FloatTensor(x_mask).to(options["device"]))

        if options["beam_decoding"]:
            for idx_s in xrange(len(test_idx)):
                if options["copy"]:
                    inputx = (torch.LongTensor(x_ext[:, idx_s]).to(options["device"]), word_emb[:, idx_s, :], dec_state[idx_s, :],\
                          torch.FloatTensor(x_mask[:, idx_s, :]).to(options["device"]), y[:, idx_s], [len_y[idx_s]], oy[idx_s],\
                          batch.max_ext_len, oovs[idx_s])
                else:
                    inputx = (torch.LongTensor(x[:, idx_s]).to(options["device"]), word_emb[:, idx_s, :], dec_state[idx_s, :],\
                          torch.FloatTensor(x_mask[:, idx_s, :]).to(options["device"]), y[:, idx_s], [len_y[idx_s]], oy[idx_s])

                beam_decode(si, inputx, model, modules, consts, options)
                si += 1
        else:
            if options["copy"]:
                inputx = (torch.LongTensor(x_ext).to(options["device"]), word_emb, dec_state, \
                          torch.FloatTensor(x_mask).to(options["device"]), y, len_y, oy, batch.max_ext_len, oovs)
            else:
                inputx = (torch.LongTensor(x).to(options["device"]), word_emb, dec_state, torch.FloatTensor(x_mask).to(options["device"]), y, len_y, oy)
            greedy_decode(test_idx, inputx, model, modules, consts, options)
            si += len(test_idx)

        testing_batch_size = len(test_idx)
        partial_num += testing_batch_size
        total_num += testing_batch_size
        if partial_num >= consts["testing_print_size"]:
            print total_num, "summs are generated"
            partial_num = 0
    pickle.dump(batches_sorted_idx, open(opt.output_dir + '/test_batch_order.pkl', 'wb'))
    print si, total_num

def run():

    all_losses = []
    p_points = []
    continuing = False
    modules, consts, options = init_modules()

    #use_gpu(consts["idx_gpu"])
    print_basic_info(modules, consts, options)

    if not opt.predict:
        print "loading train set..."
        if opt.debug:
            xy_list = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "train_small.pkl", "r"))
        else:
            xy_list = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "train.pkl", "r"))
        batch_list, num_files, num_batches = datar.batched(len(xy_list), options, consts)
        print "num_files = ", num_files, ", num_batches = ", num_batches

    running_start = time.time()
    if True: #TODO: refactor
        print('model_path', cfg.cc.MODEL_PATH)
        continue_training = len(os.listdir(cfg.cc.MODEL_PATH)) !=0
        options['continue_training'] = continue_training
        print "compiling model ..."
        model = Model(modules, consts, options)
        if options["cuda"]:
            model.cuda()

        optimizer = torch.optim.Adagrad(model.parameters(), lr=consts["lr"], initial_accumulator_value=0.1)
        existing_epoch = 0
        if continue_training or opt.predict:
            if opt.model_name == '':
                opt.model_name = list(reversed(sorted(os.listdir(cfg.cc.MODEL_PATH), key=lambda x: int(re.match('.*step(\d+)', x).groups()[0]))))[0]
                print "loading existed model:", opt.model_name
                continue_step = int(re.match('.*step(\d+)',opt.model_name).groups()[0])
            else:
                continue_step = 0
            model, optimizer, all_losses, av_batch_losses, p_points , av_batch_p_points= load_model(cfg.cc.MODEL_PATH + opt.model_name, model, optimizer)
        if opt.retrain:
            av_batch_losses = np.zeros(5)
            av_batch_p_points = np.zeros(1)
            all_losses = []
            p_points = []
            if options['coverage']:
                model.decoder.add_cov_weight()
                if options['cuda']:
                    model.cuda()
            print(model)
            if opt.retrain:
                # update optimizer, because network contains now coverage weights if coverage is on
                optimizer = torch.optim.Adagrad(model.parameters(), lr=consts["lr"], initial_accumulator_value=0.1)
            if continue_training and not opt.predict:
                continuing = True
                print('Continue training model from step {}'.format(continue_step))
        if not opt.predict:
            print "start training model "
            print_size = num_files / consts["print_time"] if num_files >= consts["print_time"] else num_files
            steps = 0
            print(model)
            # cnndm.s2s.lstm.gpu0.epoch0.7
            last_total_error = float("inf")
            print "max epoch:", consts["max_epoch"]
            for epoch in xrange(0, consts["max_epoch"]):
                print "epoch: ", epoch + existing_epoch
                num_partial = 1
                if not continuing:
                    av_batch_losses = np.zeros(5) 
                    av_batch_p_points = np.zeros(1)
                partial_num_files = 0
                epoch_start = time.time()
                partial_start = time.time()
                # shuffle the trainset
                batch_list, num_files, num_batches = datar.batched(len(xy_list), options, consts)
                used_batch = 0.
                 
                for idx_batch in xrange(num_batches):
                    if continue_training and steps <= continue_step:
                        used_batch += 1
                        init_seeds(steps)
                        steps += 1
                        partial_num_files += consts["batch_size"]
                        if partial_num_files % print_size == 0 and idx_batch < num_batches:
                            partial_num_files =0
                            num_partial += 1

                        if steps == continue_step:
                            continuing = False
                        continue
                    else:
                        continuing = False

                    train_idx = batch_list[idx_batch]
                    batch_raw = [xy_list[xy_idx] for xy_idx in train_idx]
                    if len(batch_raw) != consts["batch_size"]:
                        continue
                    local_batch_size = len(batch_raw)
                    batch = datar.get_data(batch_raw, modules, consts, options)

                    x, len_x, x_mask, y, len_y, y_mask, oy, x_ext, y_ext, oovs = sort_samples(batch.x, batch.len_x, \
                                                             batch.x_mask, batch.y, batch.len_y, batch.y_mask, \
                                                             batch.original_summarys, batch.x_ext, batch.y_ext, batch.x_ext_words)

                    model.zero_grad()

                    if opt.tf_schedule:
                        tf = teacher_forcing_ratio(steps, options["tf_offset_decay"])
                    else:
                        tf = True
                    y_pred, losses, p_point = model(torch.LongTensor(x).to(options["device"]), torch.LongTensor(len_x).to(options["device"]),\
                                   torch.LongTensor(y).to(options["device"]),  torch.FloatTensor(x_mask).to(options["device"]), \
                                   torch.FloatTensor(y_mask).to(options["device"]), torch.LongTensor(x_ext).to(options["device"]),\
                                   torch.LongTensor(y_ext).to(options["device"]), \
                                   batch.max_ext_len)
                    total_loss = 0
                    # TODO: implement averge batch costs
                    for loss_ in losses:
                        if loss_ is not None:
                            total_loss += loss_

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), consts["norm_clip"])
                    optimizer.step()

                    # append total loss to losses
                    losses = np.append(total_loss.item(), losses)

                    # transform tensors to floats
                    losses = [loss.cpu().detach().numpy() if isinstance(loss, torch.Tensor) else loss for loss in losses]

                   # with open(opt.result_path + '/result.log', "a") as log_file:
                   #     log_file.write("epoch {}, step {}, total_loss {}, loss {}, cost_cov {}, cost_p_point {}, cost_w_prior {}\n".format(epoch, steps,*losses))

                    # if new batch reset
                    # add current losses to av_batch_losses

                    av_batch_losses = np.add(av_batch_losses, losses)
                    av_batch_p_points = np.add(av_batch_p_points, p_point)
                    used_batch += 1
                    partial_num_files += consts["batch_size"]
                    if partial_num_files % print_size == 0 and idx_batch < num_batches:
                        print("Step: {}").format(steps)

                        print idx_batch + 1, "/" , num_batches, "batches have been processed,",
                        print("av_batchp_point {}, av_batch: total_loss {}, loss {}, cost_cov {}, cost_p_point {}, cost_w_prior {}".format(av_batch_p_points/used_batch, *av_batch_losses/used_batch))
                        print "time:", time.time() - partial_start
                        partial_num_files = 0
                        if not opt.debug:
                            print "save model... ",
                            save_model(cfg.cc.MODEL_PATH+ "model.gpu" + str(consts["idx_gpu"]) +".epoch"+str(epoch) + ".step"+str(steps), model, optimizer, all_losses, av_batch_losses, p_points, av_batch_p_points)
                            all_losses.append(av_batch_losses/used_batch)
                            p_points.append(av_batch_p_points/used_batch)
                            print "finished"
                        num_partial += 1
                    init_seeds(steps)
                    steps += 1

                if not continuing:
                    print("in this epoch:")
                    print("av_batchp_point {}, av_batch: total_loss {}, loss {}, cost_cov {}, cost_p_point {}, cost_w_prior {}".format(av_batch_p_points/used_batch,*av_batch_losses/used_batch))
                    print "time:", time.time() - epoch_start
            
                    print_sent_dec(y_pred, y_ext, y_mask, oovs, modules, consts, options, local_batch_size)

                    if not opt.debug:
                        print "save model... ",
                        pickle.dump([all_losses, p_points], open(opt.model_path + '/losses_p_points.p', 'wb'))
                        save_model(cfg.cc.MODEL_PATH +"model.gpu" + str(consts["idx_gpu"]) + ".epoch"+str(epoch) +  ".step" + str(steps), model, optimizer, all_losses, av_batch_losses, p_points, av_batch_p_points)
                        print "finished"

            print "save final model... ",
            save_model(cfg.cc.MODEL_PATH + "model.final.gpu" + str(consts["idx_gpu"]), model, optimizer, all_losses, av_batch_losses, p_points, av_batch_p_points)
            pickle.dump([all_losses, p_points], open(opt.model_path + '/losses_p_points.p', 'wb'))
            print "finished"
        else:
            print "skip training model"

        if opt.predict:
            predict(model, modules, consts, options)
    print "Finished, time:", time.time() - running_start

if __name__ == "__main__":
    np.set_printoptions(threshold = np.inf)
    run()
