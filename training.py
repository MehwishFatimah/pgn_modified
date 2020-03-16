#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:07:57 2019

@author: fatimamh
"""
import argparse
import os
from os import path
import sys
import time
from datetime import datetime
import math
import random
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
from torch import optim

from torch.autograd import Variable

from utils.loader_utils import get_data
from model.model_helper import * 
from utils.file_utils import *
from utils.plot_utils import showPlot
from model.model import S2SModel

import model_config as config

parser = argparse.ArgumentParser(description = 'Help for the main module')
parser.add_argument('--r', type = bool,   default = False,   help = 'To resume')
parser.add_argument('--e', type = int,   default = -1,   help = 'To resume training, provide epoch')
parser.add_argument('--f', type = str,   default = None,   help = 'To resume training, provide model')
'''----------------------------------------------------------------------------
'''
class Train(object):
    
    def __init__(self, device):
        
        self.device = device
        self.model = S2SModel(device).to(self.device)
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        
        lr = config.lr_coverage if config.coverage else config.lr
        self.optimizer = optim.Adagrad(params, lr= lr, initial_accumulator_value=config.adagrad_init_acc)
        
        sub_folder = 'train'#'train_{}'.format(int(time.time()))
        self.train_dir = os.path.join(config.log_folder, sub_folder)
        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)

        sub_folder = 'eval'#'eval_{}'.format(int(time.time()))
        self.eval_dir = os.path.join(config.log_folder, sub_folder)
        if not os.path.exists(self.eval_dir):
            os.mkdir(self.eval_dir)

    '''==============================================================================
    '''
    def save_model(self, folder, loss, epoch, b_id, prefix = 'model', suffix=None):
        
        state = {"epoch"            : epoch,
                 "batch_id"         : b_id,
                "encoder_state_dict": self.model.encoder.state_dict(),
                "decoder_state_dict": self.model.decoder.state_dict(),
                "reduce_state_dict" : self.model.reduce_state.state_dict(),
                "optimizer"         : self.optimizer.state_dict(),
                "current_loss"      : loss
        }
        if suffix is None:      
            f_name = prefix + "_e_{}_b_{}".format(epoch, b_id) + ".pth.tar"
        else:
            if prefix == 'best':
                f_name = prefix + "_model.pth.tar"
            else:
                f_name = prefix + "_e_{}_".format(epoch) + suffix + ".pth.tar"
        
        #print('f_name: {}'.format(f_name))
        file = os.path.join(folder, f_name)
        #print('file: {}'.format(file))
        torch.save(state, file)
        return file

    '''==============================================================================
    '''
    def load_model(self, file=None):
        
        print('-------------Start: load_model-------------')
    
        if file is not None and path.exists(file):
            state = torch.load(file, map_location= lambda storage, location: storage)
            s_epoch    = state["epoch"] + 1 # next epoch
            s_loss     = state["current_loss"]
            self.model.encoder.load_state_dict(state["encoder_state_dict"])
            self.model.decoder.load_state_dict(state["decoder_state_dict"])
            self.model.reduce_state.load_state_dict(state["reduce_state_dict"])
            #optimizer.load_state_dict(state["optimizer"])            
            
            if not config.coverage:
                self.optimizer.load_state_dict(state["optimizer"])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        #print('k, v: {} {}'.format(k, v))
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
        
        print('s_epoch, s_loss {} {}'.format(s_epoch, s_loss))
        print('-------------End: load_model-------------') 

        return s_epoch, s_loss

    '''==============================================================================
    '''
    def calc_running_loss(self, loss, running_loss, decay=0.99):
        
        if running_loss == 0:  # on the first iteration just take the loss
            running_loss = loss
            #print('running_loss = loss: {}'.format(running_loss))
        else:
            #print('\nrunning_loss: {} = (running_loss * decay): {} + (1 - decay) * loss: {}'.\
            #    format(running_loss, (running_loss * decay), (1 - decay) * loss))
            running_loss = (running_loss * decay) + (1 - decay) * loss 
            
        return running_loss

    '''==============================================================================
    '''
    def train_a_batch(self, input_tensor, target_tensor, input_lens, output_lens):         
        '''------------------------------------------------------------
        1: Setup tensors              
        ------------------------------------------------------------'''        
        #batch_size = input_tensor.size()[0]
        #if batch_size == 1: continue
        input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)
        '''------------------------------------------------------------
        2: Clear old gradients from the last step              
        ------------------------------------------------------------'''
        self.optimizer.zero_grad()
        '''------------------------------------------------------------
        3: Encode the sequence            
        ------------------------------------------------------------'''
        encoder_mask, context, coverage, encoder_outputs, encoder_features, encoder_hidden = self.model.encode(input_tensor, input_lens)
        #print('\tencoder_outputs: {}'.format(encoder_outputs.shape))
        #print('\tencoder_features: {}'.format(encoder_features.shape))
        encoder_hidden = self.model.state_reduction(encoder_hidden)
        '''------------------------------------------------------------
        4: Decode            
        ------------------------------------------------------------'''
        loss = self.model.decode(target_tensor, output_lens, encoder_mask, context, coverage, encoder_outputs, encoder_features, encoder_hidden)
        '''------------------------------------------------------------
        5: Compute the derivative of the loss w.r.t. the parameters 
            (or anything requiring gradients) using backpropagation.             
        ------------------------------------------------------------'''
        loss.backward()
        # clip the gradients
        norm = torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), config.grad_norm)
        torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), config.grad_norm)
        torch.nn.utils.clip_grad_norm_(self.model.reduce_state.parameters(), config.grad_norm)
        '''------------------------------------------------------------
        6: Update optimizer to take a step based on the gradients 
            of the parameters.             
        ------------------------------------------------------------'''
        self.optimizer.step()
        
        return loss.item() # loss.item() detach the loss

    '''==============================================================================
    '''
    def eval_a_batch(self, input_tensor, target_tensor, input_lens, output_lens):         
        '''------------------------------------------------------------
        1: Setup tensors              
        ------------------------------------------------------------'''        
        #batch_size = input_tensor.size()[0]
        #if batch_size == 1: continue
        input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)
        '''------------------------------------------------------------
        2: Encode the sequence            
        ------------------------------------------------------------'''
        encoder_mask, context, coverage, encoder_outputs, encoder_features, encoder_hidden = self.model.encode(input_tensor, input_lens)
        encoder_hidden = self.model.state_reduction(encoder_hidden)
        '''------------------------------------------------------------
        3: Decode            
        ------------------------------------------------------------'''
        loss = self.model.decode(target_tensor, output_lens, encoder_mask, context, coverage, encoder_outputs, encoder_features, encoder_hidden)
        
        return loss.item() # loss.item() detach the loss

    '''==============================================================================
    '''
    def start_training(self, last_epoch, epoch, train_loss, train_loader):         
        
        self.model.train()
        batch_idx = 1
        total_batch = len(train_loader)
        '''------------------------------------------------------------
        1: Get batches from loader and pass to train             
        ------------------------------------------------------------'''
        print()    
        for input_tensor, target_tensor, input_lens, output_lens in train_loader:
            print('\t---Train Batch: {}/{}---'.format(batch_idx, total_batch))

            batch_loss = self.train_a_batch(input_tensor, target_tensor, input_lens, output_lens)
            
            train_loss = self.calc_running_loss(batch_loss, train_loss)
            print('batch_loss: {}, train_loss: {}'.format(batch_loss, train_loss))
            batch_idx+=1
            if batch_idx % 5 == 0:
                f= self.save_model(self.train_dir, train_loss, epoch, batch_idx, prefix = 'train')

        if last_epoch:
            file = self.save_model(config.log_folder, train_loss, epoch, batch_idx, prefix = 'best', suffix='all')
        file = self.save_model(self.train_dir, train_loss, epoch, batch_idx, prefix = 'train', suffix='all')


        return file, train_loss

    '''==============================================================================
    '''
    def start_evaluation(self, epoch, eval_loss, best_eval_loss, eval_loader):         
        
        self.model.eval()
        batch_idx = 1
        total_batch = len(eval_loader)
        '''------------------------------------------------------------
        1: Get batches from loader and pass to evaluate             
        ------------------------------------------------------------'''
        print()
        for input_tensor, target_tensor, input_lens, output_lens in eval_loader:
            print('\t---Eval Batch: {}/{}---'.format(batch_idx, total_batch))
            
            batch_loss = self.eval_a_batch(input_tensor, target_tensor, input_lens, output_lens)
            
            eval_loss = self.calc_running_loss(batch_loss, eval_loss)
            print('batch_loss: {}, eval_loss: {}'.format(batch_loss, eval_loss))
            batch_idx+=1
            if batch_idx % 5 == 0:
                f= self.save_model(self.eval_dir, eval_loss, epoch, batch_idx, prefix ='eval')
        
        file = self.save_model(self.eval_dir, eval_loss, epoch, batch_idx, prefix = 'eval', suffix='all')
        '''------------------------------------------------------------
        2: Find best evaluation loss and save best model             
        ------------------------------------------------------------'''
        is_best = bool(eval_loss < best_eval_loss)
        best_eval_loss = (min(eval_loss, best_eval_loss))
        
        if is_best:
            file = self.save_model(self.eval_dir, eval_loss, epoch, batch_idx, prefix = 'best_model', suffix='all')

        return file, eval_loss, best_eval_loss
        
    '''==============================================================================
    '''
    def train_model(self, is_resume=False, epoch=-1, file=None):
        '''-----------------------------------------------
        Step 1: From scratch or resume
        -----------------------------------------------'''
        print(is_resume, epoch, file)
        if is_resume: # resume training
            print('resume traing')
            if epoch != -1 and file is None:
                file = "train" + "_e_{}_".format(epoch) + "all" + ".pth.tar"
            else:
                file = file
            #print(file)
            if os.path.exists(self.train_dir):    
                file = os.path.join(self.train_dir, file)
                if os.path.exists(file):
                    s_epoch, train_loss = self.load_model(file)
                    time_stamp = str(int(time.time()))
                    file = "r_"+ str(s_epoch)+ "_print_data_" + time_stamp + ".csv"
                    print_file  = os.path.join(config.out_folder, file)
                    file = "r_"+ str(s_epoch)+ "_plot_data_" + time_stamp + ".csv"
                    plot_file   = os.path.join(config.out_folder, file)
                else:
                    print('train file doesn\'t exist... {}'.format(file))
                    sys.exit()
            else:
                print('Train dir {} doesn\'t exist'.format(self.train_dir))
                sys.exit()
        
        else:
            print('start from scratch')
            s_epoch    = 1
            train_loss = 0
            time_stamp = str(int(time.time()))
            file = "print_data_" + time_stamp + ".csv"
            print_file  = os.path.join(config.out_folder, file)
            file = "plot_data_" + time_stamp + ".csv"
            plot_file   = os.path.join(config.out_folder, file)
        '''-----------------------------------------------
        Step 2: Set required variables and get from config
        -----------------------------------------------'''
        eval_loss = 0
        best_eval_loss = float('inf')
        plot_train_loss_total = 0  
        plot_eval_loss_total = 0 

        e_epoch     = config.epochs
        print_every = config.print_every
        plot_every  = config.plot_every
        #print('s_epoch: {}, e_epoch: {}, train_loss: {}, eval_loss: {}'.format(s_epoch, e_epoch, train_loss, eval_loss))
        '''-----------------------------------------------
        Step 3: Get data_loaders
        -----------------------------------------------'''
        train_loader = get_data("train")
        eval_loader   = get_data("val")        
        '''-----------------------------------------------
        Step 4: Training and evluation using train and val sets
        -----------------------------------------------'''
        last_epoch = False
        for epoch in range(s_epoch, e_epoch+1):
            print('--------Epoch:{} starts--------\n'.format(epoch))
            start_time = datetime.now()
            print('train_loss: {}, eval_loss: {}'.format(train_loss, eval_loss))
            if epoch == e_epoch:
                last_epoch = True
            file, train_loss = self.start_training(last_epoch, epoch, train_loss, train_loader)
            file, eval_loss, best_eval_loss = self.start_evaluation(epoch, eval_loss, best_eval_loss, eval_loader)            


            end_time = datetime.now()
            time_diff = get_time(start_time, end_time)
            '''------------------------------------------------------------
            5: print and plot data             
            ------------------------------------------------------------'''            
            plot_train_loss_total += train_loss
            plot_eval_loss_total += eval_loss
            
            print('\nEpoch: {}/{} ==> {:.0f}% | Time: {}'.format(epoch, e_epoch, epoch/e_epoch*100, time_diff))
            print_row = build_row(epoch, train_loss, eval_loss, best_eval_loss, time_diff)
            print_writer(print_file, print_row)

            if epoch % plot_every == 0:
                plot_train_loss_avg = plot_train_loss_total/plot_every 
                #plot_train_loss_avg/=100 # to normalize for plotting
                plot_eval_loss_avg = plot_eval_loss_total/plot_every
                #plot_eval_loss_avg/=100 # to normalize
                plot_row = build_row(epoch, plot_train_loss_avg, plot_eval_loss_avg)
                plot_writer(plot_file, plot_row)
                plot_train_loss_total = 0  
                plot_eval_loss_total = 0
            
            train_loss = 0 # reset
            eval_loss = 0
        showPlot(config.out_folder, plot_file)
        
'''==============================================================================

if __name__ == '__main__':
    
    args = parser.parse_args()
    is_resume = args.r
    epoch = args.e 
    file = args.f 
    #print (os.getcwd())
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    train = Train(device)
    train.train_model(is_resume, epoch, file)
    print(train.__dict__)
'''

