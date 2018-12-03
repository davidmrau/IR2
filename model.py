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

        self.use_p_point_loss = options["use_p_point_loss"]
        self.use_w_prior_point_loss = options["use_w_prior_point_loss"]

        self.x_ent = nn.modules.CrossEntropyLoss()

        self.eos_emb = modules["eos_emb"]
        self.lfw_emb = modules["lfw_emb"]
        self.max_len_predict = consts["max_len_predict"]
        self.min_len_predict = consts["min_len_predict"]
        self.i2w = modules["i2w"]
        self.priors = modules["priors"]
        self.priors = self.priors.to(self.device)
        self.p_point_scalar = consts["p_point_scalar"]
        self.w_prior_point_scalar = consts["w_prior_point_scalar"]
        self.dropout_p_point = consts["dropout_p_point"]

        self.dim_x = consts["dim_x"]
        self.dim_y = consts["dim_y"]
        self.len_x = consts["len_x"]
        self.len_y = consts["len_y"]
        self.hidden_size = consts["hidden_size"]
        self.n_heads = consts["n_heads"]
        self.dict_size = consts["dict_size"]
        self.pad_token_idx = consts["pad_token_idx"]
        self.ctx_size = self.hidden_size * 2 if self.is_bidirectional else self.hidden_size

        self.w_rawdata_emb = nn.Embedding(self.dict_size, self.dim_x, self.pad_token_idx)
        if self.cell == "gru":
            self.encoder = nn.GRU(self.dim_x, self.hidden_size, bidirectional=self.is_bidirectional)
            self.decoder = GRUAttentionDecoder(self.dim_y, self.hidden_size, self.ctx_size, self.device, self.copy, self.coverage, self.is_predicting)
        else:
            self.encoder = nn.LSTM(self.dim_x, self.hidden_size, bidirectional=self.is_bidirectional)
            if self.n_heads > 1:
                print "using multihead decoder"
                from lstm_dec_v1_mh_vec import LSTMAttentionDecoder
                self.decoder = LSTMAttentionDecoder(self.dim_y, self.hidden_size, self.ctx_size, self.n_heads, self.device, self.copy, self.coverage, self.is_predicting)
            else:
                print "using single head decoder"
                from lstm_dec_v1 import LSTMAttentionDecoder
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

    def decode_once(self, y, hs, dec_init_state, mask_x, x=None, max_ext_len=None, acc_att=None, mask_y=None, x_ext=None):
        batch_size = hs.size(1)
        if T.sum(y) < 0:
            y_emb = Variable(T.zeros((1, batch_size, self.dim_y))).to(self.device)
        else:
            y_emb = self.w_rawdata_emb(y)
        mask_y = Variable(T.ones((1, batch_size, 1))).to(self.device)

        if self.copy and self.coverage:
            hcs, dec_status, atted_context, att_dist, xids, C = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, xid=x_ext, init_coverage=acc_att)
        elif self.copy:
            hcs, dec_status, atted_context, att_dist, xids = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, xid=x_ext)
        elif self.coverage:
            hcs, dec_status, atted_context, att_dist, C = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, init_coverage=acc_att)
        else:
            hcs, dec_status, atted_context = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y)

        if self.copy:
            y_pred, p_point = self.word_prob(dec_status, atted_context, y_emb, att_dist, xids, max_ext_len, dropout_p_point=self.dropout_p_point)
        else:
            y_pred = self.word_prob(dec_status, atted_context, y_emb, dropout_p_point=self.dropout_p_point)

        if self.coverage and self.copy:
            return y_pred, hcs, C, att_dist, p_point
        elif self.coverage and not self.copy:
            return y_pred, hcs, C, att_dist
        else:
            return y_pred, hcs, p_point


    def forward(self, x, len_x, y, mask_x, mask_y, x_ext, y_ext, max_ext_len, tf=None):
        p_points = []
        att_dists = []
        hs, dec_init_state = self.encode(x, len_x, mask_x)

        y_emb = self.w_rawdata_emb(y)
        y_shifted = y_emb[:-1, :, :]
        y_shifted = T.cat((Variable(torch.zeros(1, *y_shifted[0].size())).to(self.device), y_shifted), 0)
        h0 = dec_init_state
        if self.cell == "lstm":
            h0 = (dec_init_state, dec_init_state)
        if self.coverage:
            acc_att = Variable(torch.zeros(T.transpose(x, 0, 1).size())).to(self.device) # B * len(x)

        if tf:
            if self.copy and self.coverage:
                hcs, dec_status, atted_context, att_dists, xids, acc_att = self.decoder(y_shifted, hs, h0, mask_x, mask_y, xid=x_ext, init_coverage=acc_att)
            elif self.copy:
                hcs, dec_status, atted_context, att_dists, xids = self.decoder(y_shifted, hs, h0, mask_x, mask_y, xid=x_ext)
            elif self.coverage:
                hcs, dec_status, atted_context, att_dists, acc_att = self.decoder(y_shifted, hs, h0, mask_x, mask_y, init_coverage=acc_att)
            else:
                hcs, dec_status, atted_context = self.decoder(y_shifted, hs, h0, mask_x, mask_y)

            if self.copy:
                y_pred, p_points = self.word_prob(dec_status, atted_context, y_shifted, att_dists, xids, max_ext_len, dropout_p_point=self.dropout_p_point)
            else:
                y_pred = self.word_prob(dec_status, atted_context, y_shifted, dropout_p_point=self.dropout_p_point)
        else:
            testing_batch_size = x.size(1)
            y_preds = []
            dec_result = [[] for i in xrange(testing_batch_size)]
            existence = [True] * testing_batch_size
            num_left = testing_batch_size
            # NOTE: why is it with -1 initialized ??
            y_shifted = torch.LongTensor(np.zeros((1, testing_batch_size), dtype="int64")).to(self.device)
            dec_state = h0

            if self.copy:
                max_steps = y_ext.size(0)
            else:
                max_steps = y.size(0)
            for step in xrange(max_steps):
                if num_left == 0:
                    break
                if self.copy and self.coverage:
                    y_pred, dec_state, acc_att, att_dist, p_point = self.decode_once(y_shifted, hs, dec_state, mask_x, x, max_ext_len, acc_att=acc_att, x_ext=x_ext)
                    # accumulate p_point for every step in the batches
                    p_points.append(p_point)
                    # accumulate attention distributions for every step in the batches
                    att_dists.append(att_dist)
                elif self.copy:
                    y_pred, dec_state, p_point = self.decode_once(y_shifted, hs, dec_state, mask_x, x, max_ext_len, x_ext=x_ext)
                    # accumulate p_point for every step in the batches
                    p_points.append(p_point)
                elif self.coverage:
                    y_pred, dec_state, acc_att, att_dist = self.decode_once(y_shifted, hs, dec_state, mask_x, acc_att=acc_att)
                    # accumulate attention distributions for every step in the batches
                    att_dists.append(att_dist)
                else:
                    y_pred, dec_state, p_point = self.decode_once(y_shifted, hs, dec_state, mask_x)
                    # accumulate p_point for every step in the batches
                    p_points.append(p_point)


                dict_size = y_pred.shape[-1]
                y_pred = y_pred.view(testing_batch_size, dict_size)
                y_preds.append(y_pred)
                next_y_ = torch.argmax(y_pred, 1)
                y_shifted = []
                for e in range(testing_batch_size):
                    eid = next_y_[e].item()
                    if eid in self.i2w:
                        y_shifted.append(eid)
                    else:
                        y_shifted.append(self.lfw_emb) # unk for copy mechanism
                y_shifted = np.array(y_shifted).reshape((1, testing_batch_size))
                y_shifted = torch.LongTensor(y_shifted).to(self.device)

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
                    if idx_max == self.eos_emb and len(dec_result[idx_doc]) >= self.min_len_predict:
                        existence[idx_doc] = False
                        num_left -= 1
                    else:
                        dec_result[idx_d/oc].append(str(idx_max))

                y_pred = torch.stack(y_preds)
        cost_p_point = 0
        cost_c = 0
        cost_w_prior_point = 0
        if (not tf) and (self.coverage or self.use_w_prior_point_loss):
            att_dists = torch.stack(att_dists)

        if self.copy:
            if not tf:
                p_points = torch.stack(p_points)
            try:
                cost = self.nll_loss(y_pred, y_ext, mask_y, self.avg_nll)
            except:
                print('Calculating the cost using the copy mechanism failed.')
                print('y_pred shape: ', y_pred.shape, ', y_ext shape: ', y_ext.shape, ', mask_y shape: ', mask_y.shape)

            if self.use_p_point_loss:
                cost_p_point = self.p_point_scalar * torch.sum(p_points.squeeze().mean(0))
            elif self.use_w_prior_point_loss:
                p_points = p_points.transpose(0,1)
                att_dists = att_dists.transpose(0,1)
                x = x.transpose(0,1)
                y_pred_idx = y_pred.transpose(0,1).argmax(2)
                cost_w_prior_point = 0
                prior_x = self.priors[x]
                for t in range(att_dists.size(1)):
                    cost_w_prior_point += (p_points[:,t]*
                    - ((prior_x* torch.log(1-att_dists[:,t]).detach()).sum())).mean()
                att_dists = att_dists.transpose(0,1)
                cost_w_prior_point = self.w_prior_point_scalar * cost_w_prior_point
        else:
            cost = self.nll_loss(y_pred, y, mask_y, self.avg_nll)

        if self.coverage:
            cost_c = T.mean(T.sum(T.min(att_dists, acc_att), 2))

        return y_pred, [cost, cost_c, cost_p_point, cost_w_prior_point], p_points.cpu().detach().view(1,-1).mean().numpy()
