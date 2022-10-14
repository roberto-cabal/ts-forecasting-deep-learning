"""
Inspired from 'Time Series Forecasting using Deep Learning' by Ivan Gridin CH3
"""

import torch.nn as nn

class GRU(nn.Module):

    def __init__(self,hidden_size,in_size=1,out_size=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size,out_size)

    def forward(self,x,h=None):
        out,_ = self.gru(x,h)
        last_hidden_states = out[:,-1]
        out = self.fn(last_hidden_states)
        return out,last_hidden_states
