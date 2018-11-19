# -*- coding: utf-8 -*-
#pylint: skip-file
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

from utils_pg import *

class WordProbLayer(nn.Module):
    def __init__(self, hidden_size, ctx_size, dim_y, dict_size, device, copy, coverage):
        super(WordProbLayer, self).__init__()
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.dim_y = dim_y
        self.dict_size = dict_size
        self.device = device
        self.copy = copy
        self.coverage = coverage

        self.w_ds = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size + self.ctx_size + self.dim_y))
        self.b_ds = nn.Parameter(torch.Tensor(self.hidden_size))
        self.w_logit = nn.Parameter(torch.Tensor(self.dict_size, self.hidden_size))
        self.b_logit = nn.Parameter(torch.Tensor(self.dict_size))

        if self.copy:
            self.v = nn.Parameter(torch.Tensor(1, self.hidden_size + self.ctx_size + self.dim_y))
            self.bv = nn.Parameter(torch.Tensor(1))

        self.init_weights()

    def init_weights(self):
        init_xavier_weight(self.w_ds)
        init_bias(self.b_ds)
        init_xavier_weight(self.w_logit)
        init_bias(self.b_logit)
        if self.copy:
            init_xavier_weight(self.v)
            init_bias(self.bv)

    def forward(self, ds, ac, y_emb, att_dist=None, xids=None, max_ext_len=None, dropout_p_point=None):
        # NOTE: why is y_emb included in the term?
        h = T.cat((ds, ac, y_emb), 2)
        logit = T.tanh(F.linear(h, self.w_ds, self.b_ds))
        logit = F.linear(logit, self.w_logit, self.b_logit)
        p_vocab = T.softmax(logit, dim = 2)

        if self.copy:
            if max_ext_len > 0:
                ext_zeros = Variable(torch.zeros(p_vocab.size(0), p_vocab.size(1), max_ext_len)).to(self.device)
                p_vocab = T.cat((p_vocab, ext_zeros), 2)
            p_gen = T.sigmoid(F.linear(h, self.v, self.bv))
            p_gen_mask_set_to_one = (torch.rand_like(p_gen) < dropout_p_point).float()
            p_gen = p_gen * (1 - p_gen_mask_set_to_one) + 0.99 * p_gen_mask_set_to_one
            p_w = (p_gen * p_vocab).scatter_add(2, xids, (1 - p_gen) * att_dist)
            return p_w, 1-p_gen
        else:
            p_w = p_vocab

        return p_w
