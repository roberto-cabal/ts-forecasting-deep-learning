"""
Inspired from 'Time Series Forecasting using Deep Learning' by Ivan Gridin CH2
"""

import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x)
        x = F.relu(x)
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

if __name__=='__main__':

    net = MnistModel()

    for name,layer in net.named_children():
        print(f'{name}: {layer}')
