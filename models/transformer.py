import torch
import torch.nn as nn
import math
from .base_model import BaseModel

class TransformerModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.embedding = nn.Linear(config["input_size"], config["hidden_size"])
        self.pos_encoder = PositionalEncoding(config["hidden_size"], config["dropout"])
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config["hidden_size"],
            nhead=4,
            dim_feedforward=4*config["hidden_size"],
            dropout=config["dropout"]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config["num_layers"])
        self.decoder = nn.Linear(config["hidden_size"], config["forecast_horizon"])
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]  # Take last time step
        return self.decoder(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)