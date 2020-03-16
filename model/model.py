#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:35:53 2020

@author: fatimamh
"""

import operator
import numpy as np
from numpy import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from model.beam import Beam
from model.encoder import Encoder
from model.reduce_state import ReduceState
from model.decoder import Decoder
import model_config as config
#from model.model_helper import *

'''---------------------------------------------------------
'''
eps = 1e-12
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

'''---------------------------------------------------------
Model class connects encoder and decoder object as 1 model
'''
class S2SModel(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        #Declare hyperparameters
        self.device         = device
        self.encoder        = Encoder(device).to(self.device)
        self.reduce_state   = ReduceState(device).to(self.device)
        self.decoder        = Decoder(device).to(self.device) 
        
        self.decoder.embedding.weight = self.encoder.embedding.weight

        #self.max_text    = config.max_text
        self.max_sum     = config.max_sum
        self.SP_index    = config.SP_index
        #self.EP_index    = config.EP_index
        self.is_coverage = config.coverage
        self.hidden_dim  = config.hid_dim

    '''---------------------------------------------------------
    '''
    def generate_mask(self, tensor):
        
        batch_size = tensor.size()[0] 
        length = tensor.size()[1]
        mask = np.zeros((batch_size, length), dtype=np.float32)
        # Fill in the numpy arrays
        for i in range(batch_size):
          for j in range(length):
            #print('tensor[i][j]: {}'.format(tensor[i][j]))
            if tensor[i][j] == 0:
                mask[i][j] = 0
            else:
                mask[i][j] = 1
            #print('mask[i][j]: {}'.format(mask[i][j]))
        return mask

    '''-------------------------------------------------------------'''
    def get_decoder_batches(self, tensor):
        
        #print('get_decoder_batches')
        #print('tensor: {}'.format(tensor.shape))
        
        batch_size = tensor.size()[0] 
        length = tensor.size()[1]

        decoder_batch = Variable(torch.zeros((batch_size, length), dtype =torch.long))
        target_batch = Variable(torch.zeros((batch_size, length), dtype =torch.long))

        for i in range(batch_size):    
            decoder_batch[i][0] = self.SP_index
            for j in range(length):
                if tensor[i][j] == 0:
                    break
                if tensor[i][j] == 3:
                    target_batch[i][j] = tensor[i][j]
                if j+1 < length:
                    if tensor[i][j] == 3:
                        decoder_batch[i][j+1] = 0
                    else:
                        decoder_batch[i][j+1] = tensor[i][j]
                        target_batch[i][j] = tensor[i][j]

            #print('tensor: {}'.format(tensor[i]))
            #print('decoder_batch: {}'.format(decoder_batch[i])) 
            #print('target_batch: {}'.format(target_batch[i]))  

        return decoder_batch, target_batch

    '''-------------------------------------------------------------'''
    def encode(self, input_tensor, input_lens):
        #print('Encoding starts')
        
        #print('input_tensor: {}'.format(input_tensor.shape))
        input_tensor = input_tensor.squeeze()
        #print('input_tensor: {}'.format(input_tensor.shape))
        
        #enc_padding_mask 
        encoder_mask = self.generate_mask(input_tensor)
        encoder_mask = torch.from_numpy(encoder_mask).to(self.device)
        #print('encoder_mask: {} '.format(encoder_mask.shape))

        batch_size = input_tensor.size()[0] 
        context = Variable(torch.zeros((batch_size, 2 * self.hidden_dim)))
        context = context.to(self.device)
        #print('context: {} '.format(context.shape))

        coverage = None
        if self.is_coverage:
            coverage = Variable(torch.zeros(input_tensor.size()))
            coverage = coverage.to(self.device)

        #if enc_batch_extend_vocab is not None:
            #enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        #if extra_zeros is not None:
            #extra_zeros = extra_zeros.cuda()

        encoder_outputs, encoder_features, encoder_hidden = self.encoder(input_tensor, input_lens)
        
        return encoder_mask, context, coverage, encoder_outputs, encoder_features, encoder_hidden
    
    '''-------------------------------------------------------------'''
    def state_reduction(self, hidden):
        
        #print('Reduction starts')
        reduced_hidden = self.reduce_state(hidden)

        return reduced_hidden
    
    '''-------------------------------------------------------------'''
    def decode(self, target_tensor, output_lens, encoder_mask, context, coverage, encoder_outputs, encoder_features, hidden):
        
        #print('Decoding starts')
        #print('traget_tensor: {}'.format(target_tensor.shape))
        target_tensor = target_tensor.squeeze()
        #print('target_tensor: {}'.format(target_tensor.shape))

        #print('output_lens: {}'.format(output_lens))
        max_batch_len = np.max(output_lens)
        max_steps = min(max_batch_len, self.max_sum)
        
        batch_size = target_tensor.size()[0] 
        
        decoder_batch, target_batch = self.get_decoder_batches(target_tensor)
        decoder_batch = decoder_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        decoder_mask = self.generate_mask(decoder_batch)
        decoder_mask = torch.from_numpy(decoder_mask).to(self.device)
        #print('decoder_mask: {} '.format(decoder_mask.shape))

        del target_tensor

        step_losses = []
        for di in range(max_steps):
            decoder_input = decoder_batch[:, di]  # Teacher forcing
            #print('decoder_input: {}'.format(decoder_input))
            
            #st1 -- encoder hidden in call - return hidden
            final_dist, hidden, context, attn_dist, p_gen, next_coverage = self.decoder(decoder_input, hidden,
                                                                                      encoder_outputs, encoder_features, 
                                                                                      encoder_mask, context, coverage, di,
                                                                                      extra_zeros =None, 
                                                                                      enc_batch_extend_vocab =None)

            target = target_batch[:, di]
            #print('target: {}'.format(target))

            #print('final_dist: {}'.format(final_dist.shape))
            #print('target.unsqueeze(1): {}'.format(target.unsqueeze(1).shape))
            
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            #print('gold_probs: {} {}'.format(gold_probs.shape, gold_probs))

            step_loss = -torch.log(gold_probs + config.eps)
            #print('step_loss: {}'.format(step_loss))

            if self.is_coverage:
                cov_loss_wt = 1.0
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                #print('coverage: {}'.format(coverage))
                
            step_mask = decoder_mask[:, di]
            #print('step_mask: {}'.format(step_mask))
            step_loss = step_loss * step_mask
            #print('step_loss: {}'.format(step_loss))
            step_losses.append(step_loss)
            #print('step_losses: {}'.format(step_losses))

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        #print('sum_losses: {}'.format(sum_losses))
        batch_avg_loss = sum_losses/batch_size
        #print('batch_avg_loss: {}'.format(batch_avg_loss))
        loss = torch.mean(batch_avg_loss)
        #print('loss: {}'.format(loss))

        return loss

    '''==============================================================================
    '''
    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    '''==============================================================================
    '''
    def beam_decode(self, input_tensor, input_lens): #batch should have only one example
        '''------------------------------------------------------------
        1: Setup tensors              
        ------------------------------------------------------------'''        
        input_tensor = input_tensor.to(self.device)
        '''------------------------------------------------------------
        2: Encode the sequence            
        ------------------------------------------------------------'''
        #print('input_tensor: {} input_lens: {}'.format(input_tensor.shape, input_lens))
        encoder_mask, context, coverage, encoder_outputs, encoder_features, encoder_hidden = self.encode(input_tensor, input_lens)
        #print('encoder_outputs: {}'.format(encoder_outputs.shape))
        #print('encoder_features: {}'.format(encoder_features.shape))
        encoder_hidden = self.state_reduction(encoder_hidden)
        '''------------------------------------------------------------
        3: setup for decode and beam            
        ------------------------------------------------------------'''
        dec_h, dec_c = encoder_hidden # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[config.SP_index], 
                      log_probs=[0.0], 
                      state=(dec_h[0], dec_c[0]), 
                      context = context[0], 
                      coverage=(coverage[0] if config.coverage else None)) for _ in range(config.beam_size)]
        
        results = []
        steps = 0
        
        while steps < config.max_sum and len(results) < config.beam_size:
        
            #print('steps: {}'.format(steps))
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < config.sum_vocab else config.UNK_index for t in latest_tokens]
            #print(latest_tokens)

            decoder_input = Variable(torch.LongTensor(latest_tokens))
            decoder_input = decoder_input.to(self.device)
            all_state_h =[]
            all_state_c = []
            all_context = []
            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            hidden_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            context_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)

                coverage_t_1 = torch.stack(all_coverage, 0)
            '''------------------------------------------------------------
            4: Decode            
            ------------------------------------------------------------'''

            final_dist, hidden, context, attn_dist, p_gen, coverage_t = self.decoder(decoder_input, hidden_1,
                                                                                      encoder_outputs, encoder_features, 
                                                                                      encoder_mask, context_1, coverage_t_1, steps,
                                                                                      extra_zeros =None, 
                                                                                      enc_batch_extend_vocab =None)
            log_probs = torch.log(final_dist)
            
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)
            
            dec_h, dec_c = hidden
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = context[i]
                coverage_i = (coverage_t[i] if config.coverage else None)
                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(), 
                                        log_prob=topk_log_probs[i, j].item(), 
                                        state=state_i, 
                                        context=context_i, 
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == config.EP_index:
                    #if steps >= config.min_dec_steps:
                    results.append(h)
                else:
                    beams.append(h)

                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)
        #print(beams_sorted[0])

        return beams_sorted[0]

    
'''       
if __name__ == '__main__':

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = S2SModel(device)
    print(model.__dict__)         
''' 


   