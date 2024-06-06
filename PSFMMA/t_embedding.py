from torch import nn
import torch


class TEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_num_layers, device):
        super(TEmbedding, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_num_layers = hidden_num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, hidden_num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.randn(self.hidden_num_layers, x.shape[0], self.hidden_size).to(self.device)
        h0.requires_grad = False
        c0 = torch.randn(self.hidden_num_layers, x.shape[0], self.hidden_size).to(self.device)
        c0.requires_grad = False
        x, (ht, ct) = self.lstm(x, (h0, c0))
        return x
