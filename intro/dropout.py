import torch
from torch.nn import Dropout
torch.manual_seed(1)

t = torch.randint(10,(5,)).float()

print(t)

dropout = Dropout(p=0.5)

dropout.train()

r = dropout(t)

print(r)

dropout.eval()

r = dropout(t)

print(r)