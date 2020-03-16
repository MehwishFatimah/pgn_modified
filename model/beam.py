"""
Created on Fri Feb 21 18:45:14 2020

@author: fatimamh
"""

import torch
from torch.autograd import Variable

'''-----------------------------------------------------------------------
'''
class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.log_probs = log_probs
        self.state = state
        self.tokens = tokens
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens = self.tokens + [token],
            log_probs = self.log_probs + [log_prob],
            state = state,
            context = context,
            coverage = coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]
    
    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)  

