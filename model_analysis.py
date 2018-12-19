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
from lstm_dec_v1_mh_vec import *
from word_prob_layer import *
import json

class Model(nn.Module):
    def __init__(self, modules, consts, options):
        super(Model, self).__init__()

        self.has_learnable_w2v = options["has_learnable_w2v"]
        self.i2w = modules['i2w']
        self.is_predicting = options["is_predicting"]
        self.is_bidirectional = options["is_bidirectional"]
        self.beam_decoding = options["beam_decoding"]
        self.cell = options["cell"]
        self.device = options["device"]
        self.copy = options["copy"]
        self.coverage = options["coverage"]
	self.w2i = modules['w2i']
        self.avg_nll = options["avg_nll"]
        self.n_heads = consts['n_heads']
        self.retrain = options['retrain']
        self.use_p_point_loss = options['use_p_point_loss']
        self.use_w_prior_point_loss = options['use_w_prior_point_loss']
        self.dim_x = consts["dim_x"]
        self.dim_y = consts["dim_y"]
        self.len_x = consts["len_x"]
        self.priors = modules['priors']
        self.p_point_scalar = consts["p_point_scalar"]
        self.w_prior_point_scalar = consts["w_prior_point_scalar"]
        self.priors = self.priors.to(self.device)
        self.len_y = consts["len_y"]
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
            self.decoder = LSTMAttentionDecoder(self.dim_y, self.hidden_size, self.ctx_size, self.n_heads, self.device, self.copy, self.coverage, self.is_predicting, self.retrain)

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

    def make_html_safe(self, s):
    	"""Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    	s.replace("<", "&lt;")
    	s.replace(">", "&gt;")
    	return s

    def show_art_oovs(self, article):
      """Returns the article string, highlighting the OOVs by placing __underscores__ around them"""
      words = article.split(' ')
      words = [("__%s__" % w) if w not in self.w2i else w for w in words]
      out_str = ' '.join(words)
      return out_str


    def show_abs_oovs(self, abstract, article_oovs):
      """Returns the abstract string, highlighting the article OOVs with __underscores__.
      If a list of article_oovs is provided, non-article OOVs are differentiated like !!__this__!!.
      Args:
	abstract: string
	vocab: Vocabulary object
	article_oovs: list of words (strings), or None (in baseline mode)
      """
      words = abstract.split(' ')
      new_words = []
      for w in words:
	if w not in self.w2i: # w is oov
	  if article_oovs is None: # baseline mode
	    new_words.append("__%s__" % w)
	  else: # pointer-generator mode
	    if w in article_oovs:
	      new_words.append("__%s__" % w)
	    else:
	      new_words.append("!!__%s__!!" % w)
	else: # w is in-vocab word
	  new_words.append(w)
      out_str = ' '.join(new_words)
      return out_str

  #self.write_for  tnvis(      None,    ref_sents, y_pred,       att_dist,  1-p_points)
    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens, oovs):
    	"""Write some data to json file, which can be read into the in-browser attention visualizer tool:
    	  https://github.com/abisee/attn_vis
    	Args:
    	  article: The original article string.
    	  abstract: The human (correct) abstract string.
    	  attn_dists: List of arrays; the attention distributions.
    	  decoded_words: List of strings; the words of the generated summary.
    	  p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
    	"""
    	print(abstract)
    	#article_lst = article.split() # list of words
	article = self.show_art_oovs(article.strip())
	abstract = self.show_abs_oovs(abstract.strip(), oovs)
    	decoded_lst = decoded_words # list of decoded words

    	to_write = {
    	    'article_lst': [self.make_html_safe(t) for t in article.split()],
    	    'decoded_lst': [self.make_html_safe(t) for t in decoded_lst],
    	    'abstract_str': self.make_html_safe(abstract),
    	    'attn_dists': [list(map(float, t[0][0])) for t in attn_dists]
    	}

	if len(attn_dists[0]) > 1:
		to_write['attn_dist_2'] = [list(map(float, t[1][0])) for t in attn_dists]
	if len(attn_dists[0]) > 2:
		to_write['attn_dist_3'] = [list(map(float, t[2][0])) for t in attn_dists]
	if len(attn_dists[0]) > 3:
		to_write['attn_dist_4'] = [list(map(float, t[3][0])) for t in attn_dists]

        if self.copy:
    	  to_write['p_gens'] = [[x] for x in p_gens]
    	output_fname = os.path.join('viz/', 'attn_vis_data.json')
    	with open(output_fname, 'w') as output_file:
    	  json.dump(to_write, output_file)
    	print('Wrote Attention vizualization')
        exit()

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
    def decode_once(self, y, hs, dec_init_state, mask_x, x=None, max_ext_len=None, acc_att=None, ref_sents=None):
        batch_size = hs.size(1)
	C = None
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
            y_pred, p_points = self.word_prob(dec_status, atted_context, y_emb, att_dist, xids, max_ext_len)
        else:
            y_pred = self.word_prob(dec_status, atted_context, y_emb)


        if self.copy or self.coverage:
            return y_pred, hcs, C, p_points.item(), att_dist.cpu().detach().numpy()

        if self.coverage:
            return y_pred, hcs, C
        else:
            return y_pred, hcs

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

        if self.copy and self.coverage:
            hcs, dec_status, atted_context, att_dist, xids, C, all_att = self.decoder(y_shifted, hs, h0, mask_x, mask_y, x_ext, acc_att)
        elif self.copy:
            hcs, dec_status, atted_context, att_dist, xids, all_att = self.decoder(y_shifted, hs, h0, mask_x, mask_y, xid=x_ext)
        elif self.coverage:
            hcs, dec_status, atted_context, att_dist, C, all_att = self.decoder(y_shifted, hs, h0, mask_x, mask_y, init_coverage=acc_att)
        else:
            hcs, dec_status, atted_context = self.decoder(y_shifted, hs, h0, mask_x, mask_y)

        if self.copy:
            y_pred, p_points = self.word_prob(dec_status, atted_context, y_shifted, att_dist, xids, max_ext_len)
        else:
            y_pred = self.word_prob(dec_status, atted_context, y_shifted)

        cost_p_point = 0
        cost_c = 0
        cost_w_prior_point = 0

        if self.copy:
            cost = self.nll_loss(y_pred, y_ext, mask_y, self.avg_nll)
            if self.use_p_point_loss:
                cost_p_point = self.p_point_scalar * torch.sum(p_points.squeeze().mean(0))
            elif self.use_w_prior_point_loss:
                p_points = p_points.transpose(0,1)
                att_dist = att_dist.transpose(0,1)
                x = x.transpose(0,1)
                y_pred_idx = y_pred.transpose(0,1).argmax(2)
                cost_w_prior_point = 0
                prior_x = self.priors[x]
                for t in range(att_dist.size(1)):
                    cost_w_prior_point += (p_points[:,t]*
                    - ((prior_x* torch.log(1-att_dist[:,t]).detach()).sum())).mean()
                att_dist = att_dist.transpose(0,1)
                cost_w_prior_point = self.w_prior_point_scalar * cost_w_prior_point
        else:
            cost = self.nll_loss(y_pred, y, mask_y, self.avg_nll)

        if self.coverage:
            cost_c = T.mean(T.sum(T.min(att_dist, acc_att), 2))

        return y_pred, [cost, cost_c, cost_p_point, cost_w_prior_point], p_points.squeeze().mean(0).mean().cpu().detach().numpy()