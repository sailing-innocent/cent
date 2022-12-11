import torch 
import torch.nn as nn 
import numpy as np
import torchvision

from torch.utils.data import DataLoader
from datetime import datetime

INPUT_SIZE = 28
HIDDEN_SIZE = 32
BATCH_SIZE = 32
EPOCH = 10
LR = 0.001
TIME_STEP = 28
DROP_RATE = 0.2
LAYERS = 2
MODEL = 'LSTM'


# log

class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()

        self.rnn = nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers = LAYERS,
            dropout = DROP_RATE,
            batch_first = True # if true, the i/o style is batch, seq_len, feature
        )

        self.hidden_out = nn.Linear(HIDDEN_SIZE, 10)
        self.h_s = None
        self.h_c = None 

    def forward(self, x):
        r_out, (h_s, h_c) = self.rnn(x) 
        output = self.hidden_out(r_out)
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

train_data = torchvision.datasets.MNIST(
    root = "./mnist",
    train = True
)
    