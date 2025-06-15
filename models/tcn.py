import torch
import torch.nn as nn
from .base_model import BaseModel

class TCNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.tcn = TemporalConvNet(
            num_inputs=config["input_size"],
            num_channels=[config["hidden_size"]] * config["num_layers"],
            kernel_size=3,
            dropout=config["dropout"]
        )
        self.linear = nn.Linear(config["hidden_size"], config["forecast_horizon"])
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len)
        y = self.tcn(x)
        y = y[:, :, -1]  # Take last time step
        return self.linear(y)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            )]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size]