#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:40:29 2020

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
class ReduceState(nn.Module):
    def __init__(self, device):
        super(ReduceState, self).__init__()

        self.device     = device
        self.hidden_dim = config.hid_dim

        self.reduce_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        init_linear_wt(self.reduce_h)
        
        self.reduce_c = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        #print('----Reduce state-----')
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        #print('RS: h: {}, c: {}'.format(h.shape, c.shape))
        
        h_in = h.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2)
        #print('RS: h_in: {}'.format(h_in.shape))
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        #print('RS: hidden_reduced_h: {}'.format(hidden_reduced_h.shape))
        hidden_reduced_h = hidden_reduced_h.unsqueeze(0)
        #print('RS: hidden_reduced_h: {}'.format(hidden_reduced_h.shape))

        c_in = c.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2)
        #print('RS: c_in: {}'.format(c_in.shape))
        hidden_reduced_c = F.relu(self.reduce_c(c_in))
        #print('RS: hidden_reduced_c: {}'.format(hidden_reduced_c.shape))
        hidden_reduced_c = hidden_reduced_c.unsqueeze(0)
        #print('-----------------')
        
        return (hidden_reduced_h, hidden_reduced_c) # h, c dim = 1 x b x hidden_dim

'''
if __name__ == '__main__':

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rs = ReduceState(device)
    print(rs.__dict__)
'''