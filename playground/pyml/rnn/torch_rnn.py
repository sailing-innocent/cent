import torch
import torch.nn as nn 

input_size = 100
hidden_size = 20
num_layers = 4

rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
print('rnn: ', rnn)

seq_len = 10
batch_size = 1
x = torch.randn(seq_len, batch_size, input_size)
h0 = torch.zeros(num_layers, batch_size, hidden_size)

out, h = rnn(x, h0)

print("out.shape: ", out.shape)
print("h.shape: ", h.shape)