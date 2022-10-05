"""
Inspired from 'Time Series Forecasting using Deep Learning' by Ivan Gridin CH3
"""

import numpy as np
from scipy import interpolate
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import torch

class DummyPredictor(torch.nn.Module):
    def forward(self,x):
        last_values = []
        l = x.tolist()
        for r in l:
            last_values.append([r[-1]])
        return torch.tensor(data=last_values)

class InterpolationPredictor(torch.nn.Module):
    def forward(self,x):
        last_values = []
        l = x.tolist()
        for r in l:
            x = np.arange(0,len(r))
            y = interpolate.interp1d(x,r,fill_value='extrapolate')
            last_values.append([y(len(r)).tolist()])
        return torch.tensor(data=last_values)

class SarimaxPredictor(torch.nn.Module):
    def forward(self,x):
        last_values = []
        l = x.tolist()
        for r in l:
            model = SARIMAX(r,order=(1,1,1),seasonal_order=(1,1,1,12))
            results = model.fit(disp=0)
            forecast = results.forecast()
            last_values.append([forecast[0]])
        return torch.tensor(data=last_values)

class HwesPredictor(torch.nn.Module):
    def forward(self,x):
        last_values = []
        l = x.tolist()
        for r in l:
            model = ExponentialSmoothing(r,trand=None,seasonal="add",seasonal_periods=12)
            results = model.fit()
            forecast = results.forecast()
            last_values.append([forecast[0]])
        return torch.tensor(data=last_values)
