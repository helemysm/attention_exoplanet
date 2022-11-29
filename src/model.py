
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math
import os
import time
import tqdm
import pandas as pd

from positional_encoding import PositionalEncoding, PositionalEncodingTime 
import logging

from box import box_from_file

logger = logging.getLogger("exoclf")

config = box_from_file(Path('config.yaml'), file_type='yaml')


class ScheduledOptim:
    def __init__(self, model_size, optimizer, lr=0.001):
        self.optimizer = optimizer
        self._step = 0
        self.model_size = model_size
        self._rate = lr
        
    def step(self, epochs):
        self._step += 1
        if epochs % 5 == 0:
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
            self.optimizer.step()
        
    def rate(self, step = None):
        
        if step is None:
            step = self._step
            self._rate = self._rate*0.8
        return self._rate

    
class ScheduledOptim_():
    
    """
    Reference: `jadore801120/attention-is-all-you-need-pytorch \
    <https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py>`_
    """

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        
        return 0.8


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    

def attention(query, key, value, mask=None, dropout=None):
    """
    Reference : https://nlp.seas.harvard.edu/2018/04/03/attention.html#hardware-and-schedule
    Compute 'Scaled Dot Product Attention
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1)h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    
class ModulesBlock(nn.Module):
    def __init__(self, layer: nn.Module, embed_dim: int, p=0.1) -> None:
        super(ModulesBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
            
        batch_size, seq_size, dim = x.size()
        device = x.device

        src =  x.transpose(0, 1)     # [seq_len, N, features]

        """
        add mask or remove if it is not necesary
        """
        mask = self.generate_square_subsequent_mask(seq_size).to(device)

        output, self.attn_weights = self.layer(src, src, src)#, attn_mask = mask)
        output = output.transpose(0, 1)     # [N, seq_len, features]

        output = self.dropout(output)
        output = self.norm(x + output)
        return output, self.attn_weights


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.fc = nn.Sequential(
        nn.Linear(hidden_size, hidden_size*2),
        #nn.ReLU(),
        nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)

        return x

    

class EncoderModule(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, dropout_rate=0.1) -> None:
        super(EncoderModule, self).__init__()
        self.attention = ModulesBlock(MultiheadAttention(embed_dim, num_head), embed_dim, dropout=dropout_rate) 
        
        #nn.MultiheadAttention(embed_dim, num_head), embed_dim, p=dropout_rate)
        
        self.feed_forward = ModulesBlock(PositionwiseFeedforwardLayer(embed_dim), embed_dim, p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, attn_weights = self.attention(x)
        x = self.feed_forward(x)
        return x, attn_weights


class FinalModule(nn.Module):
    def __init__(self, d_model: int, num_class: int, pool_size: int) -> None:
        super(FinalModule, self).__init__()
        self.d_model = d_model
        self.num_class = num_class

        self.pool_size = pool_size
        
        self.maxpool = nn.MaxPool1d(5, stride=2)
        
        #self.fc = nn.Linear(int(d_model * 2), num_class)
        self.final_layer = nn.Sequential(
            nn.Linear(int(d_model * 2), num_class),
            #nn.ReLU(),
            #nn.Linear(512, num_class)),
            nn.Sigmoid())


        #nn.init.normal_(self.final_layer.weight, std=0.02)
        #nn.init.normal_(self.final_layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(-1, int(self.d_model*2))
        
        x = self.maxpool(x)
        
        #x = self.fc(x)
        x = self.final_layer(x) 
        return x


    
class EncoderLayer(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=conf.modelparam.emb_dim, dropout_rate=0.2) -> None:
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.seq_size = seq_len
        
        self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        self.positional_encoding = PositionalEncodingTime(d_model, None, seq_len)
        
        embedding_size = d_model
        
        self.position_embedding = nn.Embedding(num_embeddings = seq_len, embedding_dim = d_model)
        
        self.modules = nn.ModuleList([
            EncoderModule(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def generate_positional_sequence(self, sz, seq_sz):
        position = torch.arange(0, seq_sz, dtype=torch.int64).unsqueeze(0).repeat(sz, 1)
        return position
 
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, x_cen: torch.Tensor, x_time) -> torch.Tensor:
        
        #  1
        #**********************
        # cat flux and centroid
        x = torch.cat([x, x_cen], -1)
        x = x.transpose(1, 2)
        
        #  2
        #**********************
        #x_cen = x_cen.transpose(1, 2)
        #x_cen = self.input_embedding(x_cen)
        #x_cen = x_cen.transpose(1, 2)
        #**********************
        
        x = self.input_embedding(x) # cnn embedding
        x = x.transpose(1, 2)
        
        #  2
        #**********************
        #x_emb = torch.cat([x, x_cen], -1)
        #device = x_emb.device
        #batch_size, seq_size, dim = x_emb.size()
        #x_emb = self.positional_encoding(x_emb)
        #x = x_emb
        #**********************
        
        device = x.device
        batch_size, seq_size, dim = x.size()
        #x = self.positional_encoding(x)
        pe = PositionalEncodingTime(512, seq_len, x_time)
        x = pe(x)
        
        all_attn_weights = []
        
        for l in self.modules:
            x , attn_weights= l(x)
            all_attn_weights.append(np.array(attn_weights.cpu().detach()))
        
        
        all_attn_weights_copy = all_attn_weights.copy() 
        all_attn_weights_copy = torch.Tensor(np.array(all_attn_weights_copy))
        
        all_attn_weights_copy = all_attn_weights_copy.transpose(0,1)
        
        
        return x, all_attn_weights_copy

    
    

class classification_model(nn.Module):

    def __init__(
            self, input_features: int, seq_len: int, n_heads: int,
            n_class: int, n_layers: int, d_model: int = 512, dropout_rate: float = 0.2
    ) -> None:
        super(classification_model, self).__init__()
        self.encoder = EncoderLayer(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.encoder_global = EncoderLayer(input_features, seq_len*4, n_heads, n_layers, d_model, dropout_rate)
        
        #stellar/transit
        input_features_stell = 1
        seq_len_stell = 6
        d_model_stell = 32
        
        self.embb_stellar = nn.Conv1d(input_features_stell, d_model_stell, 1)
        self.blocks_stell = nn.ModuleList([
            EncoderBlock(d_model_stell, n_heads, dropout_rate) for _ in range(n_layers)
        ])
        
        self.clf = FinalModule(d_model, n_class)

    def forward(self, x: torch.Tensor, x_cen: torch.Tensor,  x_global: torch.Tensor, x_cen_global: torch.Tensor, x_stell: torch.Tensor, x_timel, x_timeg) -> torch.Tensor:
        
        x_stell = x_stell.transpose(1, 2)
        
        x_stell = self.embb_stellar(x_stell)
        x_stell = x_stell.transpose(1, 2)
        
        all_attn_weights_stell = []
        
        for l in self.blocks_stell:
            x_stell , attn_weights_stell= l(x_stell)
            all_attn_weights_stell.append(np.array(attn_weights_stell.cpu().detach()))
        
        all_attn_weights_stell_copy = all_attn_weights_stell.copy() 
        all_attn_weights_stell_copy = torch.Tensor(np.array(all_attn_weights_stell_copy))
        
        all_attn_weights_stell_copy = all_attn_weights_stell_copy.transpose(0,1)
        
        
        x = torch.cat([x, x_cen], -1)
        x = x.transpose(1, 2)
        x, attn_weights = self.encoder(x, x_timel)
        
        x_global = torch.cat([x_global, x_cen_global], -1)
        x_global = x_global.transpose(1, 2)
        x_global, attn_weights_global = self.encoder_global(x_global, x_timeg)
        
        out_local = x.reshape(x.shape[0], -1)
        out_global = x_global.reshape(x_global.shape[0], -1)
        
        out_stell = x_stell.reshape(x_stell.shape[0], -1)
        
        #cat
        out = torch.cat([out_local, out_global, out_stell], dim=1)
        
        
        x = self.clf(out)
        
        return x, out_local,out_global, attn_weights, attn_weights_global, all_attn_weights_stell_copy
    
    


