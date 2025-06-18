import torch
import torch.nn as nn
from .base_model import BaseModel

class CNN_LSTM_Model(BaseModel):
    """
    A hybrid CNN-LSTM model for time series forecasting.
    
    The CNN layer processes subsequences to extract features, and the LSTM layer
    models the temporal relationships between these features.
    """
    def __init__(self, config):
        super(CNN_LSTM_Model, self).__init__(config)
        
        # Unpack model-specific parameters from the config dictionary
        input_size = self.config['input_size']
        cnn_out_channels = self.config['cnn_out_channels']
        cnn_kernel_size = self.config['cnn_kernel_size']
        lstm_hidden_size = self.config['lstm_hidden_size']
        lstm_num_layers = self.config['lstm_num_layers']
        dropout = self.config['dropout']
        output_size = self.config['forecast_horizon']

        # 1. CNN Feature Extractor
        # We apply a 1D convolution over the sequence length.
        # The input has 'input_size' channels (e.g., 1 for univariate).
        self.cnn = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            padding='same' # 'same' padding ensures the output sequence length is the same as input
        )
        self.relu = nn.ReLU()

        # 2. LSTM for Temporal Modeling
        # The input to the LSTM is the feature map from the CNN.
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

        # 3. Fully Connected Output Layer
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, input_size)
        
        # --- CNN Part ---
        # Conv1d expects input of shape (batch_size, channels, sequence_length)
        # So we permute the last two dimensions.
        x_cnn_in = x.permute(0, 2, 1)
        
        cnn_out = self.cnn(x_cnn_in)
        cnn_out = self.relu(cnn_out)
        
        # The output of Conv1d is (batch_size, cnn_out_channels, sequence_length).
        # We need to permute it back for the LSTM.
        x_lstm_in = cnn_out.permute(0, 2, 1)
        
        # --- LSTM Part ---
        # Input to LSTM is now (batch_size, sequence_length, cnn_out_channels)
        lstm_out, _ = self.lstm(x_lstm_in)
        
        # We only need the output of the last time step for forecasting
        last_time_step_out = lstm_out[:, -1, :]
        
        # --- Output Layer ---
        out = self.fc(last_time_step_out)
        
        return out
