

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import math
import os
import time
import tqdm
import pandas as pd

import logging

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x) -> torch.Tensor:
        seq_len = x.shape[1]
        x = math.sqrt(self.d_model) * x
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        return x



class PositionalEncodingTime(nn.Module):
    def __init__(self, d_model, seq_len, time, dropout=0.1, max_len=5000):
            
    PE = np.zeros((max_len, d_model))
    pos = time[:, np.newaxis]
    #pos = timefolded[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(pos * div_term)
    PE[:, 1::2] = np.cos(pos * div_term)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

    
def subsequent_mask(size: int) -> torch.Tensor:

    attn_shape = (size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("float32")
    mask = torch.from_numpy(mask) == 0
    return mask.float()
