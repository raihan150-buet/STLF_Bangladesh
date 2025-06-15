import torch
import torch.nn as nn
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.lstm = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"] if config["num_layers"] > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(config["dropout"])
        self.linear = nn.Linear(config["hidden_size"], config["forecast_horizon"])
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        predictions = self.linear(last_out)
        return predictions