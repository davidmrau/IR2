import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from utils_pg import *

class LSTMAttentionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, ctx_size, n_heads, device, copy, coverage, is_predicting):
        super(LSTMAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.is_predicting = is_predicting
        self.device = device
        self.copy = copy
        self.coverage = coverage 
        self.n_heads = n_heads
        
        self.lstm_1 = nn.LSTMCell(self.input_size, self.hidden_size)

        self.Wx = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.ctx_size))
        self.Ux = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.hidden_size))
        self.bx = nn.Parameter(torch.Tensor(4 * self.hidden_size))

        self.Wc_att = nn.ParameterList()
        self.b_att = nn.ParameterList()
        self.W_comb_att = nn.ParameterList()
        self.U_att = nn.ParameterList()
        self.W_coverage = nn.ParameterList()
        self.W_att_concat = nn.Linear(self.ctx_size * self.n_heads, self.ctx_size, bias=False)

        for i in range(self.n_heads):
            self.Wc_att.append(nn.Parameter(torch.Tensor(self.ctx_size, self.ctx_size)))
            self.b_att.append(nn.Parameter(torch.Tensor(self.ctx_size)))

            self.W_comb_att.append(nn.Parameter(torch.Tensor(self.ctx_size, 2*self.hidden_size)))
            self.U_att.append(nn.Parameter(torch.Tensor(1, self.ctx_size)))

            if self.coverage:
                self.W_coverage.append(nn.Parameter(torch.Tensor(self.ctx_size, 1)))

        self.init_weights()

    def init_weights(self):
        init_lstm_weight(self.lstm_1)
        init_ortho_weight(self.Wx)
        init_ortho_weight(self.Ux)
        init_bias(self.bx)
        for i in range(self.n_heads):
            init_ortho_weight(self.Wc_att[i])
            init_bias(self.b_att[i])
            init_ortho_weight(self.W_comb_att[i])
            init_ortho_weight(self.U_att[i])
            if self.coverage:
                init_ortho_weight(self.W_coverage[i])


    def forward(self, y_emb, context, init_state, x_mask, y_mask, xid=None, init_coverage=None):
        def _get_word_atten(pctx, h1, x_mask, i, acc_att=None): #acc_att: B * len(x)
            if acc_att is not None:
                h = F.linear(h1, self.W_comb_att[i]) + F.linear(T.transpose(acc_att, 0, 1).unsqueeze(2), self.W_coverage[i]) # len(x) * B * ?
            else:
                h = F.linear(h1, self.W_comb_att[i])
            unreg_att = T.tanh(pctx[i] + h) * x_mask
            unreg_att = F.linear(unreg_att, self.U_att[i])

            word_atten = T.exp(unreg_att - T.max(unreg_att, 0, keepdim=True)[0]) * x_mask
            sum_word_atten = T.sum(word_atten, 0, keepdim=True)
            word_atten =  word_atten / sum_word_atten
            return word_atten

        def recurrence(x, y_mask, hidden, pctx, context, x_mask, acc_att=None):
            pre_h, pre_c = hidden

            h1, c1 = self.lstm_1(x, hidden)  
            h1 = y_mask * h1 + (1.0 - y_mask) * pre_h
            c1 = y_mask * c1 + (1.0 - y_mask) * pre_c
            
            # len(x) * batch_size * 1
            s = T.cat((h1.view(-1, self.hidden_size), c1.view(-1, self.hidden_size)), 1)
            if self.coverage:
                word_atten = [_get_word_atten(pctx, s, x_mask, i, acc_att) for i in range(self.n_heads)]
            else:
                word_atten = [_get_word_atten(pctx, s, x_mask, i) for i in range(self.n_heads)]
            atted_ctx_heads = [T.sum(word_atten[i] * context, 0) for i in range(self.n_heads)]
            atted_ctx_concat = T.cat(atted_ctx, -1)
            atted_ctx = self.W_att_concat(atted_ctx_concat)



            ifoc_preact = F.linear(h1, self.Ux) + F.linear(atted_ctx, self.Wx, self.bx)
            x4i, x4f, x4o, x4c = ifoc_preact.chunk(4, 1)
            i = torch.sigmoid(x4i)
            f = torch.sigmoid(x4f)
            o = torch.sigmoid(x4o)
            c2 = f * c1 + i * torch.tanh(x4c)
            h2 = o * torch.tanh(c2)
            c2 = y_mask * c2 + (1.0 - y_mask) * c1
            h2 = y_mask * h2 + (1.0 - y_mask) * h1

            word_atten_point = T.transpose(word_atten[0].view(x_mask.size(0), -1), 0, 1)

            word_atten_avg = torch.mean(torch.stack(word_atten, 0), dim=0)
            word_atten_avg = T.transpose(word_atten_avg.view(x_mask.size(0), -1), 0, 1)
            
            if self.coverage:
                acc_att += word_atten_avg
                return (h2, c2), h2, atted_ctx, word_atten_point, acc_att
            else:
                return (h2, c2), h2, atted_ctx, word_atten_point

        hs, cs, ss, atts, dists, xids, Cs = [], [], [], [], [], [], []
        hidden = init_state
        acc_att = init_coverage
        if self.copy: 
            xid = T.transpose(xid, 0, 1) # B * len(x)

        pctx = [F.linear(context, self.Wc_att[i], self.b_att[i]) for i in range(self.n_heads)]
        x = y_emb
        
        steps = range(y_emb.size(0))
        for i in steps:
            if self.coverage:
                Cs += [acc_att]
                hidden, s, att, att_dist, acc_att = recurrence(x[i], y_mask[i], hidden, pctx, context, x_mask, acc_att)
            else:
                hidden, s, att, att_dist = recurrence(x[i], y_mask[i], hidden, pctx,context, x_mask)
            hs += [hidden[0]]
            cs += [hidden[1]]
            ss += [s]
            atts += [att]
            dists += [att_dist]
            xids += [xid]
        
        if self.coverage:
            if self.is_predicting :
                Cs += [acc_att]
                Cs = Cs[1:]
            Cs = T.stack(Cs).view(y_emb.size(0), *Cs[0].size())
        

        hs = T.stack(hs).view(y_emb.size(0), *hs[0].size())
        cs = T.stack(cs).view(y_emb.size(0), *cs[0].size())
        ss = T.stack(ss).view(y_emb.size(0), *ss[0].size())
        atts = T.stack(atts).view(y_emb.size(0), *atts[0].size())
        dists = T.stack(dists).view(y_emb.size(0), *dists[0].size())
        if self.copy:
            xids = T.stack(xids).view(y_emb.size(0), *xids[0].size())
        
        if self.copy and self.coverage:
            return (hs, cs), ss, atts, dists, xids, Cs
        elif self.copy:
            return (hs, cs), ss, atts, dists, xids
        elif self.coverage:
            return (hs, cs), ss, atts, dists, Cs
        else:
            return (hs, cs), ss, atts
