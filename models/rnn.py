import random
import numpy as np
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 src_input_size, tgt_input_size,
                 rnn_type, device, d_r, seed):

        super(RNN, self).__init__()
        self.enc_lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.dec_lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.linear_enc = nn.Linear(src_input_size, hidden_size, bias=False)
        self.linear_dec = nn.Linear(tgt_input_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)
        self.device = device
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def forward(self, x_en, x_de, hidden=None):

        x_en = self.linear_enc(x_en).permute(1, 0, 2)
        x_de = self.linear_dec(x_de).permute(1, 0, 2)

        if hidden is None:
            hidden = torch.zeros(self.n_layers, x_en.shape[1], self.hidden_size).to(self.device)

        _, (hidden, state) = self.enc_lstm(x_en, (hidden, hidden))
        dec_output, _ = self.dec_lstm(x_de, (hidden, hidden))

        outputs = self.linear2(dec_output).transpose(0, 1)

        return outputs