import torch.nn as nn
from .base_model import BaseModel

class ClassicalConvLSTM(BaseModel):
    """
    A purely classical counterpart to the QuantumInspiredLSTM.
    
    This model replaces the special 'QIConv1D' layer with a standard 'nn.Conv1d'
    layer, allowing for a direct benchmark of the quantum-inspired technique.
    """
    def __init__(self, config):
        super(ClassicalConvLSTM, self).__init__(config)
        
        # Unpack parameters from the config
        self.lstm_hidden_size = self.config['lstm_hidden_size']
        self.lstm_num_layers = self.config['lstm_num_layers']
        
        # Parameters for the standard convolutional layer
        conv_out_channels = self.config['conv_out_channels']
        conv_kernel_size = self.config['conv_kernel_size']
        
        # 1. Standard LSTM layer (Identical to the quantum-inspired model)
        self.lstm = nn.LSTM(
            input_size=self.config['input_size'],
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=self.config['dropout'] if self.lstm_num_layers > 1 else 0
        )
        
        # 2. Standard 1D Convolutional Layer
        # This is the classical equivalent that replaces the QIConv1D layer.
        self.conv1d = nn.Conv1d(
            in_channels=self.lstm_hidden_size,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            padding='same' # Keep the sequence length the same
        )
        self.relu = nn.ReLU()
        
        # 3. Final Output Layer
        self.fc = nn.Linear(conv_out_channels, self.config['forecast_horizon'])

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, input_size)
        
        # Pass through the LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, lstm_hidden_size)
        
        # Permute for the 1D convolution layer
        # Conv1d expects: (batch_size, channels, sequence_length)
        conv_in = lstm_out.permute(0, 2, 1)
        
        # Pass through the standard Convolution
        conv_out = self.conv1d(conv_in)
        conv_out = self.relu(conv_out)
        
        # Take the output of the last time step for our final prediction
        final_features = conv_out[:, :, -1]
        
        # Pass through the final fully-connected layer
        out = self.fc(final_features)
        
        return out
