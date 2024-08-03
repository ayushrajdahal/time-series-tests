import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
 
torch.set_default_dtype(torch.float64)
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# XLSTM Cell
class XLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(XLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wc = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)
        self.We = nn.Linear(input_size + hidden_size, hidden_size)
 
    def forward(self, x, h_prev, c_prev):
        combined = torch.cat((x, h_prev), dim=1)
        ft = torch.sigmoid(self.Wf(combined))
        it = torch.sigmoid(self.Wi(combined))
        ct_hat = torch.tanh(self.Wc(combined))
        ot = torch.sigmoid(self.Wo(combined))
        et = torch.sigmoid(self.We(combined))
        ct = ft * c_prev + it * ct_hat
        ht = ot * torch.tanh(ct)
        ht = et * torch.exp(ht) + (1 - et) * ht
        return ht, ct
 
# Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
 
    def forward(self, lstm_out):
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return context_vector, attention_weights
 
# XLSTM with Attention Model for Solar Power Forecasting
class XLSTM_Attention_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(XLSTM_Attention_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.xlstm_cells = nn.ModuleList([XLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
       
        lstm_out = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i in range(self.num_layers):
                if i == 0:
                    h[i], c[i] = self.xlstm_cells[i](x_t, h[i], c[i])
                else:
                    h[i], c[i] = self.xlstm_cells[i](h[i-1], h[i], c[i])
            lstm_out.append(h[-1])
       
        lstm_out = torch.stack(lstm_out, dim=1)
        context_vector, attention_weights = self.attention(lstm_out)
        out = self.fc(context_vector)
        return out, attention_weights
