import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel

class QIConv1D(nn.Module):
    """
    A Quantum-Inspired 1D Convolutional Layer.
    
    This layer adapts the mathematical principles of the 2D version for 1D time series data.
    It treats inputs and weights as angles and performs a convolution that is
    analogous to complex number multiplication, inspired by quantum mechanics.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(QIConv1D, self).__init__()
        
        # The learnable parameters are angles (theta), not direct weights.
        self.cnn_weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size))
        self.bias_weight = nn.Parameter(torch.rand(out_channels))
        
        self.stride = stride
        self.padding = padding

    def forward(self, inputs):
        # inputs are expected to be (batch, channels, seq_len)
        
        # Decompose inputs and weights into sin/cos components (like real/imaginary parts)
        cos_inputs = torch.cos(inputs)
        sin_inputs = torch.sin(inputs)
        cos_theta = torch.cos(self.cnn_weight)
        sin_theta = torch.sin(self.cnn_weight)
        real_bias = torch.cos(self.bias_weight)
        imag_bias = torch.sin(self.bias_weight)

        # Perform 1D convolutions that mimic complex number multiplication
        real = F.conv1d(cos_inputs, cos_theta, bias=real_bias, padding=self.padding, stride=self.stride) - \
               F.conv1d(sin_inputs, sin_theta, bias=real_bias, padding=self.padding, stride=self.stride)
        
        imag = F.conv1d(sin_inputs, cos_theta, bias=imag_bias, padding=self.padding, stride=self.stride) + \
               F.conv1d(cos_inputs, sin_theta, bias=imag_bias, padding=self.padding, stride=self.stride)

        # The output is the phase angle of the resulting complex number
        y = (np.pi / 2) - torch.atan2(imag, real)
        
        return y

class QuantumInspiredLSTM(BaseModel):
    """
    A hybrid model that uses a standard LSTM followed by a Quantum-Inspired
    1D Convolutional layer to refine the learned features.
    """
    def __init__(self, config):
        super(QuantumInspiredLSTM, self).__init__(config)
        
        # Unpack parameters from the config
        self.lstm_hidden_size = self.config['lstm_hidden_size']
        self.lstm_num_layers = self.config['lstm_num_layers']
        qi_conv_out_channels = self.config['qi_conv_out_channels']
        qi_conv_kernel_size = self.config['qi_conv_kernel_size']
        
        # 1. Standard LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.config['input_size'],
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=self.config['dropout'] if self.lstm_num_layers > 1 else 0
        )
        
        # 2. Quantum-Inspired Convolutional Layer
        # This layer will process the sequence of hidden states from the LSTM
        self.qi_conv = QIConv1D(
            in_channels=self.lstm_hidden_size,
            out_channels=qi_conv_out_channels,
            kernel_size=qi_conv_kernel_size
        )
        self.relu = nn.ReLU()
        
        # 3. Final Output Layer
        # This layer takes the refined features from the QI-Conv layer and makes a prediction.
        self.fc = nn.Linear(qi_conv_out_channels, self.config['forecast_horizon'])

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, input_size)
        
        # Pass through the LSTM
        # We need the full sequence of outputs, not just the last hidden state
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, lstm_hidden_size)
        
        # Permute for the 1D convolution layer
        # Conv1d expects: (batch_size, channels, sequence_length)
        # Here, the lstm_hidden_size acts as the "channels"
        conv_in = lstm_out.permute(0, 2, 1)
        
        # Pass through the Quantum-Inspired Convolution
        conv_out = self.qi_conv(conv_in)
        conv_out = self.relu(conv_out)
        
        # We take the output of the last time step for our final prediction
        # conv_out shape: (batch_size, qi_conv_out_channels, sequence_length)
        final_features = conv_out[:, :, -1]
        
        # Pass through the final fully-connected layer
        out = self.fc(final_features)
        
        return out
