import torch
import torch.nn as nn
from .base_model import BaseModel

class ClassicalResLSTM(BaseModel):
    """
    A purely classical benchmark for the QResLSTM.
    It replaces the quantum residual block with a classical MLP (Multi-Layer Perceptron).
    """
    def __init__(self, config):
        super(ClassicalResLSTM, self).__init__(config)
        
        # Identical LSTM backbone
        self.lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config.get('dropout', 0.1) if config['num_layers'] > 1 else 0
        )
        
        # Classical replacement for quantum residual (a small MLP)
        self.residual_processor = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size'] // 2),
            nn.GELU(), # GELU is a smooth, high-performing activation function
            nn.Linear(config['hidden_size'] // 2, config['hidden_size']),
            nn.LayerNorm(config['hidden_size'])
        )
        
        # The final prediction layer
        self.fc = nn.Linear(config['hidden_size'], config['forecast_horizon'])
        
        # Store the q_delta value to ensure the scaling of the residual is identical
        self.q_delta = config.get('q_delta', 0.05)

    def forward(self, x):
        # 1. Classical LSTM processing
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # 2. Classical residual calculation
        residual = self.residual_processor(last_hidden)
        
        # 3. Apply the residual connection with the same scaling factor
        hybrid_features = last_hidden + residual * self.q_delta
        
        # 4. Final prediction
        return self.fc(hybrid_features)
