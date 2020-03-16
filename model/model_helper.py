#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:40:29 2020

@author: fatimamh
"""

import os
import sys
import time
from datetime import datetime
from numpy import random

import torch
import torch.nn as nn
import torch.optim as optim
import model_config as config
'''---------------------------------------------------------
'''
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

#rand_unif_init_mag=0.02
#trunc_norm_init_std=1e-4

'''---------------------------------------------------------
'''
def init_lstm_wt(lstm):
    
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, 
                                  config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

'''---------------------------------------------------------
'''
def init_linear_wt(linear):
    
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

'''---------------------------------------------------------
'''
def init_wt_normal(wt):
    
    wt.data.normal_(std=config.trunc_norm_init_std)

'''---------------------------------------------------------
'''
def init_wt_unif(wt):
    
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

'''----------------------------------------------------------------
'''
def get_time(st, et):
    
    diff = str('{}d:{}h:{}m:{}s'.\
           format(et.day-st.day,
           et.hour-st.hour,
           et.minute-st.minute,
           et.second-st.second))

    return diff

'''----------------------------------------------------------------
'''
# TO DO: VErify
def total_params(model):
    for parameter in model.parameters():
            print(parameter.size(), len(parameter)) 
            print()
'''----------------------------------------------------------------
'''
def trainable_params(model):     
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
        
    print('params: {}'.format(params))