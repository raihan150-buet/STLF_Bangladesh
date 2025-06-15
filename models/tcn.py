import torch
import torch.nn as nn
from .base_model import BaseModel

class Chomp1d(nn.Module):
    """A helper module to remove padding from the end of a sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """A single block of a Temporal Convolutional Network."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
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

class TemporalConvNet(nn.Module):
    """The full stack of TemporalBlocks that makes up the TCN."""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
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

class TCNModel(BaseModel):
    """The main TCN model class that wraps the TemporalConvNet."""
    def __init__(self, config):
        super(TCNModel, self).__init__()
        # Unpack the config dictionary to build the TCN
        self.tcn = TemporalConvNet(
            num_inputs=config["input_size"],
            num_channels=[config["hidden_size"]] * config["num_layers"],
            kernel_size=config.get("kernel_size", 3), # Use kernel_size from config or default
            dropout=config["dropout"]
        )
        self.linear = nn.Linear(config["hidden_size"], config["forecast_horizon"])

    def forward(self, x):
        # TCN expects input of shape (batch_size, num_features, sequence_length)
        x_permuted = x.permute(0, 2, 1)
        tcn_out = self.tcn(x_permuted)
        
        # We take the output of the very last time step from the TCN
        last_time_step_out = tcn_out[:, :, -1]
        
        # Pass it through the final fully connected layer
        return self.linear(last_time_step_out)