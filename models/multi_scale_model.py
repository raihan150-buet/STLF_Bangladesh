import torch
import torch.nn as nn
from .base_model import BaseModel

class MultiScaleModel(BaseModel):
    """
    A multi-scale parallel model that processes the time series at different
    sequence lengths simultaneously to capture short, medium, and long-term patterns.
    """
    def __init__(self, config):
        super(MultiScaleModel, self).__init__(config)
        self.config = config
        input_size = config['input_size']

        # --- Define sequence lengths for each branch ---
        self.cnn_seq_len = config['cnn_seq_len']
        self.gru_seq_len = config['gru_seq_len']
        self.lstm_seq_len = config['lstm_seq_len']

        # --- Branch 1: Short-Term (CNN) ---
        cnn_params = config['cnn_params']
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels=input_size,
                      out_channels=cnn_params['out_channels'],
                      kernel_size=cnn_params['kernel_size'],
                      padding='same'),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate the flattened output size for the CNN branch
        cnn_output_size = cnn_params['out_channels'] * self.cnn_seq_len

        # --- Branch 2: Medium-Term (GRU) ---
        gru_params = config['gru_params']
        self.gru_branch = nn.GRU(input_size=input_size,
                                 hidden_size=gru_params['hidden_size'],
                                 num_layers=gru_params['num_layers'],
                                 batch_first=True)

        # --- Branch 3: Long-Term (LSTM) ---
        lstm_params = config['lstm_params']
        self.lstm_branch = nn.LSTM(input_size=input_size,
                                   hidden_size=lstm_params['hidden_size'],
                                   num_layers=lstm_params['num_layers'],
                                   batch_first=True)

        # --- Decision Fusion Layer ---
        fusion_input_size = cnn_output_size + gru_params['hidden_size'] + lstm_params['hidden_size']
        fusion_params = config['fusion_params']
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_params['hidden_size']),
            nn.ReLU(),
            nn.Dropout(fusion_params.get('dropout', 0.1)),
            nn.Linear(fusion_params['hidden_size'], config['forecast_horizon'])
        )

    def forward(self, x):
        # Ensure the input tensor is on the correct device
        device = next(self.parameters()).device
        x = x.to(device)

        # Slice input for each branch
        x_cnn = x[:, -self.cnn_seq_len:, :]
        x_gru = x[:, -self.gru_seq_len:, :]
        x_lstm = x
        
        # CNN Branch (already produces 2D output)
        cnn_out = self.cnn_branch(x_cnn.permute(0, 2, 1))

        # GRU Branch
        _, gru_hidden_state = self.gru_branch(x_gru)
        gru_out = gru_hidden_state[-1, :, :]

        # LSTM Branch
        _, (lstm_hidden_state, _) = self.lstm_branch(x_lstm)
        lstm_out = lstm_hidden_state[-1, :, :]

        # --- Concatenate the 2D outputs from all branches ---
        combined_out = torch.cat((cnn_out, gru_out, lstm_out), dim=1)

        # Final prediction from the fusion layer
        final_prediction = self.fusion_layer(combined_out)

        # Squeeze to match target shape
        return final_prediction.squeeze(-1)

        
