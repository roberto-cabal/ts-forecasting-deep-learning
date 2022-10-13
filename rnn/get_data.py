"""
Functions extracted directly from 'Time Series Forecasting using Deep Learning' by Ivan Gridin CH2
"""

import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

def get_aep_timeseries():
    df = pd.read_csv(os.environ['PATH_TO_DATA']+'AEP_hourly.csv')
    # df['Datetime'] = pd.to_datetime(df['Datetime'])
    # df = df.sort_values(by='Datetime',ignore_index=True)
    ts = df['AEP_MW'].astype(int).values.reshape(-1, 1)[-3000:]
    return ts

def sliding_window(ts, features):
    X = []
    Y = []

    for i in range(features + 1, len(ts) + 1):
        X.append(ts[i - (features + 1):i - 1])
        Y.append([ts[i - 1]])

    return X, Y

def get_training_datasets(ts, features, test_len):
    X, Y = sliding_window(ts, features)

    X_train, Y_train, X_test, Y_test = X[0:-test_len],\
                                       Y[0:-test_len],\
                                       X[-test_len:],\
                                       Y[-test_len:]

    train_len = round(len(ts) * 0.7)

    X_train, X_val, Y_train, Y_val = X_train[0:train_len],\
                                     X_train[train_len:],\
                                     Y_train[0:train_len],\
                                     Y_train[train_len:]

    x_train = torch.tensor(data = X_train).float()
    y_train = torch.tensor(data = Y_train).float()

    x_val = torch.tensor(data = X_val).float()
    y_val = torch.tensor(data = Y_val).float()

    x_test = torch.tensor(data = X_test).float()
    y_test = torch.tensor(data = Y_test).float()

    return x_train, x_val, x_test, y_train.squeeze(1), y_val.squeeze(1), y_test.squeeze(1)
