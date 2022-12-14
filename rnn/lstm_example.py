import pandas as pd
import matplotlib.pyplot as plt
import copy
import random
import sys
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from lstm import LSTM
from get_data import *
from competing_models import *

# seed
random.seed(1)
torch.manual_seed(1)

# --- parameters
features = 240
test_ts_len = 300
lstm_hidden_size = 24
learning_rate = 0.02
training_epochs = 500

# --- read data
ts = get_aep_timeseries()
# plt.title('AEP Hourly')
# plt.plot(ts[:500])
# plt.show()

# --- scale data
scaler = MinMaxScaler()
scaled_ts = scaler.fit_transform(ts) # TODO: SPLIT BEFORE SCALE
x_train, x_val, x_test, y_train, y_val, y_test = get_training_datasets(scaled_ts, features, test_ts_len)
#import pdb; pdb.set_trace()

# --- define model
model = LSTM(hidden_size = lstm_hidden_size)
model.train()

# --- ALTERNATIVE MODELS
dummy_predictor = DummyPredictor()
sarima_predictor = SarimaxPredictor()
hwes_predictor = HwesPredictor()

yhat_dummy = dummy_predictor(x_test)
yhat_sarima = sarima_predictor(x_test)
yhat_hwes = hwes_predictor(x_test)

yhat_dummy = scaler.inverse_transform(yhat_dummy.reshape(-1,1)).reshape(-1)
yhat_sarima = scaler.inverse_transform(yhat_sarima.reshape(-1,1)).reshape(-1)
yhat_hwes = scaler.inverse_transform(yhat_hwes.reshape(-1,1)).reshape(-1)

real = scaler.inverse_transform(y_test.tolist()).reshape(-1)
ms_error_dummy = np.mean((real-yhat_dummy)**2)
ms_error_sarima = np.mean((real-yhat_sarima)**2)
ms_error_hwes = np.mean((real-yhat_hwes)**2)

# --- training
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
mse_loss = torch.nn.MSELoss()

best_model = None

min_val_loss = sys.maxsize
training_loss = []
validation_loss = []

for t in range(training_epochs):

    prediction, _ = model(x_train)
    loss = mse_loss(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    val_prediction, _ = model(x_val)
    val_loss = mse_loss(val_prediction, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())

    if val_loss.item() < min_val_loss:
        best_model = copy.deepcopy(model)
        min_val_loss = val_loss.item()

    if t % 50 == 0:
        print(f'epoch {t}: train - {round(loss.item(), 4)}, '
              f'val: - {round(val_loss.item(), 4)}')

# --- predict on test set
use_warm_up = False

if use_warm_up:
    best_model.eval()
    with torch.no_grad():
        _, h_list = best_model(x_val)
        # warm hidden state
        h = tuple([(h[-1,-1,:]).unsqueeze(-2).unsqueeze(-2) for h in h_list])

predicted = []
for test_seq in x_test.tolist():
    x = torch.Tensor(data = [test_seq])
    # passing hidden state and cell through each iteration
    if use_warm_up:
        y, h = best_model(x, h)
    else:
        y,_ = best_model(x)
    unscaled = scaler.inverse_transform(np.array(y.item()).reshape(-1, 1))[0][0]
    predicted.append(unscaled)

real = scaler.inverse_transform(y_test.tolist()).reshape(-1)
ms_error = np.mean((real-np.array(predicted))**2)

# --- training progress
plt.figure()
plt.title('Training')
plt.yscale('log')
plt.plot(training_loss, label = 'training')
plt.plot(validation_loss, label = 'validation')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

msse = ms_error/ms_error_dummy
plt.figure()
plt.title(f"Test dataset: MSE = {ms_error}, MSSE = {msse}")
plt.plot(real,label='real')
plt.plot(predicted,label='predicted')
plt.plot(yhat_dummy,label='dummy')
plt.plot(yhat_sarima,label='sarima')
plt.plot(yhat_hwes,label='hwes')
plt.legend()
plt.show()

print(f'LSTM MSE = {ms_error}')
print(f'Dummy MSE = {ms_error_dummy}')
print(f'Sarima MSE = {ms_error_sarima}')
print(f'Hwes MSE = {ms_error_hwes}')
