#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:17:28 2020

@author: hsuankai
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Body Movement Network')
    
    # global arguments
    parser.add_argument('--data', type=str, default='../Data/all/train_mfcc.pkl', help='if true use mel-spectrogram else use mfcc as feature')
    parser.add_argument('--delay', type=int, default=0, help='time-delay')
    parser.add_argument('--epoch', type=int, default=300, help='epoch')
    parser.add_argument('--batch', type=int, default=32, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--early_stop_iter', type=int, default=10, help='use early stopping scheme if > 0')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/warmup/', help='checkpoint')
    
    # model arguments
    parser.add_argument('--d_input', type=int, default=28, help='the hidden units used in network')
    parser.add_argument('--d_model', type=int, default=512, help='the hidden units used in network')
    parser.add_argument('--d_output_body', type=int, default=39, help='the number of body points')
    parser.add_argument('--d_output_rh', type=int, default=6, help='the number of right-hand points')
    parser.add_argument('--warmup_steps', type=int, default=500, help='warmup steps')
    parser.add_argument('--n_block', type=int, default=2, help='the number of u-net block')
    parser.add_argument('--n_unet', type=int, default=4, help='the number of u-net layer')
    parser.add_argument('--n_attn', type=int, default=1, help='the number of self-attention layer')
    parser.add_argument('--n_head', type=int, default=4, help='the number of head')
    parser.add_argument('--max_len', type=int, default=900, help='the max length of sequence')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--pre_lnorm', type=bool, default=False, help='apply pre-layernormalization or not')
    parser.add_argument('--attn_type', type=str, default='rel', help='the type of self-attention') 
    
    args = parser.parse_args()
    
    return args