"""
Inspired from 'Time Series Forecasting using Deep Learning' by Ivan Gridin CH3
"""

import torch.nn as nn

"""
Inspired from 'Time Series Forecasting using Deep Learning' by Ivan Gridin CH3
"""

import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self,hidden_size,in_size=1,out_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fn = nn.Linear(hidden_size,out_size)

    def forward(self,x,h=None):
        out,h = self.lstm(x,h)
        last_hidden_states = out[:,-1]
        out = self.fn(last_hidden_states)
        return out,h
