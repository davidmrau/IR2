# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import numpy as np
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable

from utils_pg import *
from gru_dec import *
from lstm_dec_v2 import *
from word_prob_layer import *
import random

class Model(nn.Module):
    def __init__(self, modules, consts, options):
        super(Model, self).__init__()

        self.has_learnable_w2v = options["has_learnable_w2v"]
        self.is_predicting = options["is_predicting"]
        self.is_bidirectional = options["is_bidirectional"]
        self.beam_decoding = options["beam_decoding"]
        self.cell = options["cell"]
        self.device = options["device"]
        self.copy = options["copy"]
        self.coverage = options["coverage"]
        self.avg_nll = options["avg_nll"]

        self.dim_x = consts["dim_x"]
        self.dim_y = consts["dim_y"]
        self.len_x = consts["len_x"]
        self.len_y = consts["len_y"]
        self.teacher_forcing_p = consts["teacher_forcing_p"]
        self.hidden_size = consts["hidden_size"]
        self.dict_size = consts["dict_size"]
        self.pad_token_idx = consts["pad_token_idx"]
        self.ctx_size = self.hidden_size * 2 if self.is_bidirectional else self.hidden_size

        self.w_rawdata_emb = nn.Embedding(self.dict_size, self.dim_x, self.pad_token_idx)
        if self.cell == "gru":
            self.encoder = nn.GRU(self.dim_x, self.hidden_size, bidirectional=self.is_bidirectional)
            self.decoder = GRUAttentionDecoder(self.dim_y, self.hidden_size, self.ctx_size, self.device, self.copy, self.coverage, self.is_predicting)
        else:
            self.encoder = nn.LSTM(self.dim_x, self.hidden_size, bidirectional=self.is_bidirectional)
            self.decoder = LSTMAttentionDecoder(self.dim_y, self.hidden_size, self.ctx_size, self.device, self.copy, self.coverage, self.is_predicting)

        self.get_dec_init_state = nn.Linear(self.ctx_size, self.hidden_size)
        self.word_prob = WordProbLayer(self.hidden_size, self.ctx_size, self.dim_y, self.dict_size, self.device, self.copy, self.coverage)

        self.init_weights()

    def init_weights(self):
        init_uniform_weight(self.w_rawdata_emb.weight)
        if self.cell == "gru":
            init_gru_weight(self.encoder)
        else:
            init_lstm_weight(self.encoder)
        init_linear_weight(self.get_dec_init_state)

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -T.log(T.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = T.sum(cost * y_mask, 0) / T.sum(y_mask, 0)
        else:
            cost = T.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return T.mean(cost)

    def encode(self, x, len_x, mask_x):
        self.encoder.flatten_parameters()
        emb_x = self.w_rawdata_emb(x)

        emb_x = torch.nn.utils.rnn.pack_padded_sequence(emb_x, len_x)
        hs, hn = self.encoder(emb_x, None)
        hs, _ = torch.nn.utils.rnn.pad_packed_sequence(hs)

        dec_init_state = T.sum(hs * mask_x, 0) / T.sum(mask_x, 0)
        dec_init_state = T.tanh(self.get_dec_init_state(dec_init_state))
        return hs, dec_init_state

    def decode_once(self, y, hs, dec_init_state, mask_x, x=None, max_ext_len=None, acc_att=None):
        batch_size = hs.size(1)
        if T.sum(y) < 0:
            y_emb = Variable(T.zeros((1, batch_size, self.dim_y))).to(self.device)
        else:
            y_emb = self.w_rawdata_emb(y)
        mask_y = Variable(T.ones((1, batch_size, 1))).to(self.device)

        if self.copy and self.coverage:
            hcs, dec_status, atted_context, att_dist, xids, C = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, x, acc_att)
        elif self.copy:
            hcs, dec_status, atted_context, att_dist, xids = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, xid=x)
        elif self.coverage:
            hcs, dec_status, atted_context, att_dist, C = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, init_coverage=acc_att)
        else:
            hcs, dec_status, atted_context = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y)

        if self.copy:
            y_pred = self.word_prob(dec_status, atted_context, y_emb, att_dist, xids, max_ext_len)
        else:
            y_pred = self.word_prob(dec_status, atted_context, y_emb)

        if self.coverage:
            return y_pred, hcs, C
        else:
            return y_pred, hcs
def greedy_decode(flist, batch, model, modules, consts, options, mask_y):
    testing_batch_size = len(flist)

    dec_result = [[] for i in xrange(testing_batch_size)]
    existence = [True] * testing_batch_size
    num_left = testing_batch_size

    if self.copy
        x, word_emb, dec_state, x_mask, y, len_y, ref_sents, max_ext_len, oovs = batch
    else:
        x, word_emb, dec_state, x_mask, y, len_y, ref_sents = batch

    next_y = torch.LongTensor(-np.ones((1, testing_batch_size), dtype="int64")).to(options["device"])

    if self.cell == "lstm":
        dec_state = (dec_state, dec_state)
    if self.coverage:
        acc_att = Variable(torch.zeros(T.transpose(x, 0, 1).size())).to(options["device"]) # B *len(x)

    for step in xrange(consts["max_len_predict"]):
        if num_left == 0:
            break
        if self.copy and self.coverage:
            y_pred, dec_state, acc_att = model.decode_once(next_y, word_emb, dec_state, x_mask, x, max_ext_len, acc_att)
        elif self.copy:
            y_pred, dec_state = model.decode_once(next_y, word_emb, dec_state, x_mask, x, max_ext_len)
        elif self.coverage:
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

        if self.coverage:
            acc_att = acc_att.view(testing_batch_size, acc_att.shape[-1])

        if self.cell == "lstm":
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

        cost = self.nll_loss(y_pred, y, mask_y, self.avg_nll)


    def forward(self, x, len_x, y, mask_x, mask_y, x_ext, y_ext, max_ext_len):

        hs, dec_init_state = self.encode(x, len_x, mask_x)

        y_emb = self.w_rawdata_emb(y)
        y_shifted = y_emb[:-1, :, :]
        y_shifted = T.cat((Variable(torch.zeros(1, *y_shifted[0].size())).to(self.device), y_shifted), 0)
        h0 = dec_init_state
        if self.cell == "lstm":
            h0 = (dec_init_state, dec_init_state)
        if self.coverage:
            acc_att = Variable(torch.zeros(T.transpose(x, 0, 1).size())).to(self.device) # B * len(x)

        if random.random() > self.teacher_forcing_p:
            if self.copy and self.coverage:
                hcs, dec_status, atted_context, att_dist, xids, C = self.decoder(y_shifted, hs, h0, mask_x, mask_y, x_ext, acc_att)
            elif self.copy:
                hcs, dec_status, atted_context, att_dist, xids = self.decoder(y_shifted, hs, h0, mask_x, mask_y, xid=x_ext)
            elif self.coverage:
                hcs, dec_status, atted_context, att_dist, C = self.decoder(y_shifted, hs, h0, mask_x, mask_y, init_coverage=acc_att)
            else:
                hcs, dec_status, atted_context = self.decoder(y_shifted, hs, h0, mask_x, mask_y)

            if self.copy:
                y_pred = self.word_prob(dec_status, atted_context, y_shifted, att_dist, xids, max_ext_len)
            else:
                y_pred = self.word_prob(dec_status, atted_context, y_shifted)
        else:
            testing_batch_size = x.size(0)
            y_preds = []
            dec_result = [[] for i in xrange(testing_batch_size)]
            existence = [True] * testing_batch_size
            num_left = testing_batch_size

            y_shifted = torch.LongTensor(-np.ones((1, testing_batch_size), dtype="int64")).to(options["device"])

            for step in xrange(consts["max_len_predict"]):
                if num_left == 0:
                    break
                if self.copy and self.coverage:
                    y_pred, dec_state, acc_att = model.decode_once(y_shifted, word_emb, dec_state, x_mask, x, max_ext_len, acc_att)
                elif self.copy:
                    y_pred, dec_state = model.decode_once(y_shifted, word_emb, dec_state, x_mask, x, max_ext_len)
                elif self.coverage:
                    y_pred, dec_state, acc_att = model.decode_once(y_shifted, word_emb, dec_state, x_mask, acc_att=acc_att)
                else:
                    y_pred, dec_state = model.decode_once(y_shifted, word_emb, dec_state, x_mask)

                y_preds.append(y_pred)

                dict_size = y_pred.shape[-1]
                y_pred = y_pred.view(testing_batch_size, dict_size)
                next_y_ = torch.argmax(y_pred, 1)
                y_shifted = []
                for e in range(testing_batch_size):
                    eid = next_y_[e].item()
                    if eid in modules["i2w"]:
                        y_shifted.append(eid)
                    else:
                        y_shifted.append(modules["lfw_emb"]) # unk for copy mechanism
                y_shifted = np.array(y_shifted).reshape((1, testing_batch_size))
                y_shifted = torch.LongTensor(y_shifted).to(options["device"])

                if self.coverage:
                    acc_att = acc_att.view(testing_batch_size, acc_att.shape[-1])

                if self.cell == "lstm":
                    dec_state = (dec_state[0].view(testing_batch_size, dec_state[0].shape[-1]), dec_state[1].view(testing_batch_size, dec_state[1].shape[-1]))
                else:
                    dec_state = dec_state.view(testing_batch_size, dec_state.shape[-1])

                for idx_doc in xrange(testing_batch_size):
                    if existence[idx_doc] == False:
                        continue

                    idx_max = y_shifted[0, idx_doc].item()
                    if idx_max == modules["eos_emb"] and len(dec_result[idx_doc]) >= consts["min_len_predict"]:
                        existence[idx_doc] = False
                        num_left -= 1
                    else:
                        dec_result[idx_doc].append(str(idx_max))

                y_pred = torch.stack(y_preds)
        if self.copy:
            cost = self.nll_loss(y_pred, y_ext, mask_y, self.avg_nll)
        else:
            cost = self.nll_loss(y_pred, y, mask_y, self.avg_nll)

        if self.coverage:
            cost_c = T.mean(T.sum(T.min(att_dist, C), 2))
            return y_pred, cost, cost_c
        else:
            return y_pred, cost, None
