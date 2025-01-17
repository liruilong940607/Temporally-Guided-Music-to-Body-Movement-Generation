import torch
import torch.nn as nn

from .module import  Linear
from .attention import FFN_linear
from .layers import Unet_block


class HandEncoder(nn.Module):
    """
    HandEncoder with Self-attention and U-net
    """
    def __init__(self, d_input, d_model, n_block, n_unet, n_attn, n_head, max_len, dropout,
                 pre_lnorm, attn_type):
        super(HandEncoder, self).__init__()
        self.linear = Linear(d_input, d_model)
        self.unet = nn.ModuleList([Unet_block(d_model, n_unet, n_attn, n_head, max_len, dropout, pre_lnorm, attn_type)] * n_block)
        self.ffn = FFN_linear(d_model, dropout)
             
    def forward(self, enc_input, lengths, return_attns=False):
        x = self.linear(enc_input)
        for unet in self.unet:
            x = unet(x, lengths, return_attns)      
        enc_output = self.ffn(x)
        return enc_output
        
class Generator(nn.Module):
    """
    LSTM Generator
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(Generator, self).__init__()       
        self.output_dim = output_dim
        
        # Trainable h & c
        h_init = \
            nn.init.constant_(torch.empty(1, 1, hidden_dim), 0.0)
        c_init = \
            nn.init.constant_(torch.empty(1, 1, hidden_dim), 0.0)
        self.h_init = nn.Parameter(h_init, requires_grad=True)
        self.c_init = nn.Parameter(c_init, requires_grad=True)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.fc = Linear(hidden_dim, output_dim)        

        self.initialize()

    def initialize(self):
        # Initialize LSTM Weights and Biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    torch.nn.init.normal_(weight.data, 0.0, 0.02)
                else:
                    bias = getattr(self.lstm, param_name)
                    nn.init.constant_(bias.data, 0.0)

    def forward(self, inputs, lengths):
        batch_size = inputs.size(0)
        total_length = inputs.size(1)
        
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True)
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(inputs, (self.h_init.repeat(1, batch_size, 1), self.c_init.repeat(1, batch_size, 1)))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=total_length)

        output = output.contiguous()
        output = output.view(-1, output.size(-1))  # flatten before FC
        output = self.dropout(output)
        output = self.fc(output)
        output = output.view(-1, total_length, self.output_dim)
        return output

class MovementNet(nn.Module):
    """
    Full Body Movement Network
    """
    def __init__(self, d_input, d_output_body, d_output_rh, d_model, n_block, n_unet, n_attn, n_head,  max_len, dropout, 
                 pre_layernorm=False, attn_type='rel', gpu='0'):
        super(MovementNet, self).__init__()
        self.gpu = gpu
        
        if d_output_body > 0:
            self.bodynet = Generator(d_input, d_model, d_output_body, dropout)
        else:
            self.bodynet = None
        self.handencoder = HandEncoder(d_input, d_model, n_block, n_unet, n_attn, n_head, max_len, dropout, 
                                       pre_layernorm, attn_type)
        self.handdecoder = Generator(d_model, d_model, d_output_rh, dropout)
        self.refine_network = Linear(d_model, 3)
         
    def forward(self, inputs, lengths, return_attns=False):
        """
        Args: 
            inputs: [B, T, D]
            lengths: [B]  
            
        Returns:
            output: [B, T, (K*3)]
        """
        if self.bodynet:
            body_output = self.bodynet(inputs, lengths)        
            enc_output = self.handencoder.forward(inputs, lengths, return_attns=return_attns)
            rh_output = self.handdecoder.forward(enc_output, lengths)
            rh_refined = self.refine_network(enc_output)
            rh_refined = rh_output[:, :, -3:] + rh_refined  # root
            rh_final = torch.cat([rh_output[:, :, :-3], rh_refined], dim=-1)
            full_output = torch.cat([body_output, rh_final], dim=-1)
        else: # AIST++ dataset
            enc_output = self.handencoder.forward(inputs, lengths, return_attns=return_attns)
            output = self.handdecoder.forward(enc_output, lengths)
            refined = self.refine_network(enc_output)
            refined = output[:, :, :3] + refined  # root
            full_output = torch.cat([refined, output[:, :, 3:]], dim=-1)
        return full_output
        
        
        
        
        