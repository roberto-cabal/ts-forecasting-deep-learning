"""
Inspired from 'Time Series Forecasting using Deep Learning' by Ivan Gridin CH2
"""

import copy
from pickletools import optimize
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time_series_data import get_time_series_datasets
from fcnn_model import FCNN
import random

random.seed(42)
torch.manual_seed(42)

features = 256
ts_len = 3000

x_train,x_val,x_test,y_train,y_val,y_test = get_time_series_datasets(features,ts_len)

net = FCNN(n_inp=features,l_1=64,l_2=32,n_out=1)
net.train()

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=net.parameters())

best_model = None
min_val_loss = 1000000

for t in range(10000):
    prediction = net(x_train)
    loss = loss_func(prediction,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    val_prediction = net(x_val)
    val_loss = loss_func(val_prediction,y_val)
    if val_loss.item()<min_val_loss:
        best_model = copy.deepcopy(net)
        min_val_loss = val_loss.item()
    if t%1000 == 0:
        print(f'epoch {t}: train - {round(loss.item(),4)}, val - {round(val_loss.item(),4)}')

print('TESTING')
print(f'FCNN Loss: {loss_func(best_model(x_test),y_test).item()}')
