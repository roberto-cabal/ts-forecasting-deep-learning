"""
Inspired from 'Time Series Forecasting using Deep Learning' by Ivan Gridin CH3
"""

import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim

class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

    def forward(self,x):
        flat = x.view(x.shape[0],x.shape[1],self.input_size)
        out,h = self.lstm(flat)
        return out,h

class Decoder(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1,num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
        self.linear = nn.Linear(hidden_size,output_size)

    def forward(self,x,h):
        out,h = self.lstm(x.unsqueeze(0),h)
        y = self.linear(out.squeeze(0))
        return y,h
