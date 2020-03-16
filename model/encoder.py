"""
Created on Fri Feb 21 18:45:14 2020

@author: fatimamh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.model_helper import *
import model_config as config

'''-----------------------------------------------------------------------
'''
class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()

        # Declare the hyperparameter
        self.device     = device
        self.input_dim  = config.text_vocab
        self.embed_dim  = config.emb_dim
        self.hidden_dim = config.hid_dim
        self.n_layers   = config.n_layers
        
        self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
        init_wt_normal(self.embedding.weight)
         
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers= self.n_layers, 
                            batch_first= False, bidirectional=True) #batch_first=True,
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)    

    def forward(self, input, input_lens):
        #print('---------------------------')
        
        #print('Encoder:\tinput: {}'.format(input.shape))
        
        #batch_size = input.size()[0]       
        embedded = self.embedding(input)
        #print('Encoder:\tembedding: {}'.format(embedded.shape))
        
        packed = pack_padded_sequence(embedded, input_lens, batch_first=True, enforce_sorted= False) # throws error if enforce_sorted is not false
        #print('packed: {}'.format(packed.data.shape))
        encoder_outputs, hidden = self.lstm(packed)    
        #print('Encoder:\tgru-output: {}\n\t\t\tgru-hidden: {}'.format(encoder_outputs.shape, hidden.shape))
        
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()
        #print('encoder_outputs: {}'.format(encoder_outputs.shape))
        #if encoder_outputs.size()[1] != config.max_text:

        encoder_feature = encoder_outputs.view(-1, 2*self.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)
        #print('Encoder:\tfeature: {}'.format(encoder_feature.shape))
        #print('---------------------------')
        
        return encoder_outputs, encoder_feature, hidden  

'''
if __name__ == '__main__':

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(device)
    print(encoder.__dict__)
'''