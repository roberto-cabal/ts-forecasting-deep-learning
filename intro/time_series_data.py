"""
Functions extracted directly from 'Time Series Forecasting using Deep Learning' by Ivan Gridin CH2
"""

import random
from math import sin, cos
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

def get_time_series_data(length):
    a = .2
    b = 300
    c = 20
    ls = 5
    ms = 20
    gs = 100

    ts = []

    for i in range(length):
        ts.append(b + a * i + ls * sin(i / 5) + ms * cos(i / 24) + gs * sin(i / 120) + c * random.random())

    return ts

def get_time_series_datasets(features, ts_len):
    ts = get_time_series_data(ts_len)

    X = []
    Y = []
    for i in range(features + 1, ts_len):
        X.append(ts[i - (features + 1):i - 1])
        Y.append([ts[i]])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, shuffle = False)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.5, shuffle = False)

    x_train = torch.tensor(data = X_train)
    y_train = torch.tensor(data = Y_train)

    x_val = torch.tensor(data = X_val)
    y_val = torch.tensor(data = Y_val)

    x_test = torch.tensor(data = X_test)
    y_test = torch.tensor(data = Y_test)

    return x_train, x_val, x_test, y_train, y_val, y_test

if __name__ == '__main__':
    # data = get_time_series_data(3_000)
    # plt.figure()
    # plt.plot(data)
    # plt.show()

    features = 256
    ts_len = 3000

    _,_,_,y_train,y_val,y_test = get_time_series_datasets(features,ts_len)

    plt.figure()
    plt.plot(range(len(y_train)),y_train,label='train')
    plt.plot(range(len(y_train),len(y_train)+len(y_val)),y_val,label='val')
    plt.plot(range(len(y_train)+len(y_val),len(y_train)+len(y_val)+len(y_test)),y_test,label='test')
    plt.legend()
    plt.show()
