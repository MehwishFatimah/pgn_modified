#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:45:14 2020

@author: fatimamh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.model_helper import *
import model_config as config
'''-----------------------------------------------------------------------
'''
class Attention(nn.Module):
    def __init__(self, is_coverage, hidden_dim):
        super(Attention, self).__init__()
        #self.device      = device
        self.is_coverage = is_coverage
        self.hidden_dim  = hidden_dim
        
        if self.is_coverage:
            self.W_c = nn.Linear(1, self.hidden_dim * 2, bias=False)
        
        self.decode_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.v = nn.Linear(self.hidden_dim * 2, 1, bias=False)
    
    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if self.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, self.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if self.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

'''-----------------------------------------------------------------------
'''
class Decoder(nn.Module):
    
    def __init__(self, device):
        super(Decoder, self).__init__()

        # Declare the hyperparameter
        self.device       = device
        self.is_coverage  = config.coverage
        self.pgn          = config.pgn
        self.emb_dim      = config.emb_dim
        self.hidden_dim   = config.hid_dim
        self.output_dim   = config.sum_vocab
        self.n_layers     = config.n_layers
        
        
        # Define layers
        self.attention_network = Attention(self.is_coverage, self.hidden_dim)

        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        init_wt_normal(self.embedding.weight)
                
        self.x_context = nn.Linear(self.hidden_dim * 2 + self.emb_dim, self.emb_dim)
        
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_layers, 
                            batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)    
               
        if self.pgn:
            self.p_gen_linear = nn.Linear(self.hidden_dim * 4 + self.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.out2 = nn.Linear(self.hidden_dim, self.output_dim)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, coverage, step, extra_zeros =None, enc_batch_extend_vocab =None):

        #print('---------decoder----------')
        #print('self.training: {}, step: {}'.format(self.training, step))
        #print('if not self.training and step == 0 {}'.format(not self.training and step == 0))
        
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            #print('h_decoder: {}, c_decoder: {}'.format(h_decoder.shape, c_decoder.shape))

            s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
                                 c_decoder.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                              enc_padding_mask, coverage)

            #print('s_t_hat: {}, c_t: {}, coverage_next: {}'.format(s_t_hat.shape, c_t.shape, coverage_next.shape))
            coverage = coverage_next
            #print('coverage: {}'.format(coverage.shape))

        y_t_1_embd = self.embedding(y_t_1)
        #print('y_t_1_embd: {}'.format(y_t_1_embd.shape))

        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        #print('x: {}'.format(x.shape))
        x = x.unsqueeze(1)
        #print('x: {}'.format(x.shape))
        
        lstm_out, s_t = self.lstm(x, s_t_1)
        #print('after lstm layer')
        h_decoder, c_decoder = s_t
        #print('h_decoder: {}, c_decoder: {}'.format(h_decoder.shape, c_decoder.shape))

        s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
                             c_decoder.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
        #print('s_t_hat: {}'.format(s_t_hat.shape))

        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                          enc_padding_mask, coverage)
        #print('c_t: {}, attn_dist: {}, coverage_next: {}'.format(c_t.shape, attn_dist.shape, coverage_next))


        #print('self.training: {}, step: {}'.format(self.training, step))
        #print('self.training or step > 0 {}'.format(self.training or step > 0))
        if self.training or step > 0:
            coverage = coverage_next
            #print('coverage: {}'.format(coverage)

        if self.pgn:
            #print('c_t: {}'.format(c_t.shape))
            #print('s_t_hat: {}'.format(s_t_hat.shape))
            #print('x: {}'.format(x.squeeze().shape))
            p_gen_input = torch.cat((c_t, s_t_hat, x.squeeze()), 1)  # B x (2*2*hidden_dim + emb_dim)
            #print('p_gen_input: {}'.format(p_gen_input.shape))

            p_gen = self.p_gen_linear(p_gen_input)
            #print('p_gen: {}'.format(p_gen.shape))

            p_gen = torch.sigmoid(p_gen)
            #print('p_gen: {}'.format(p_gen.shape))

        output = torch.cat((lstm_out.view(-1, self.hidden_dim), c_t), 1) # B x hidden_dim * 3
        #print('output: {}'.format(output.shape))
        output = self.out1(output) # B x hidden_dim
        #print('output: {}'.format(output.shape))
        
        output = self.out2(output) # B x vocab_size
        #print('output: {}'.format(output.shape))

        vocab_dist = F.softmax(output, dim=1)
        #print('vocab_dist: {}'.format(vocab_dist.shape))

        if self.pgn:
            vocab_dist_ = p_gen * vocab_dist
            #print('vocab_dist: {}'.format(vocab_dist.shape))

            attn_dist_ = (1 - p_gen) * attn_dist
            #print('attn_dist_: {}'.format(attn_dist_.shape))

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
                #print('vocab_dist: {}'.format(vocab_dist.shape))
            
            if enc_batch_extend_vocab is not None:
                final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
            else:
                final_dist = vocab_dist
            #print('final_dist: {}'.format(final_dist.shape))
        else:
            final_dist = vocab_dist
            #print('final_dist: {}'.format(final_dist.shape))

        #print('---------------------')
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

'''
if __name__ == '__main__':

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    decoder = Decoder(device)
    print(decoder.__dict__)
'''