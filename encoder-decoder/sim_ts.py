"""
Inspired from 'Time Series Forecasting using Deep Learning' by Ivan Gridin CH5
"""

import numpy as np
import torch

def generate_ts(ts_len):
    tf = 80*np.pi
    t = np.linspace(0.0,tf,ts_len)
    y = np.sin(t) + 0.8*np.cos(5*t) + np.random.normal(0,0.3,ts_len) + 2.5
    return y.tolist()

def sliding_window(ts, features, target_len = 1):
    X = []
    Y = []

    for i in range(features + target_len, len(ts) + 1):
        X.append(ts[i - (features + target_len):i - target_len])
        Y.append(ts[i - target_len:i])

    return X, Y

def to_tensor(data):
    return torch.tensor(data=data).unsqueeze(2).transpose(0,1).float()

if __name__=='__main__':

    ts_len = 2000
    ts_history_len = 240
    ts_target_len = 60
    test_ds_len = 200

    ts = generate_ts(ts_len)
    X,Y = sliding_window(ts,ts_history_len,ts_target_len)
    ds_len = len(X)
    print(np.array(X).shape)
    print(np.array(X))
    print(np.array(Y).shape)
    print(np.array(Y))

    x_train = to_tensor(X[:ds_len-test_ds_len])
    y_train = to_tensor(Y[:ds_len-test_ds_len])
    x_test = to_tensor(X[ds_len-test_ds_len:])
    y_test = to_tensor(Y[ds_len-test_ds_len:])

    print(ds_len-test_ds_len)
    print(x_train.shape)
    print(x_train)
    print(y_train.shape)
    print(y_train)
