import torch
import torch.nn as nn
import math
from .base_model import BaseModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class QuantumResidualTransformer(BaseModel):
    def __init__(self, config):
        super(QuantumResidualTransformer, self).__init__(config)
        
        self.d_model = config['d_model']
        self.use_quantum_features = config.get('use_quantum_features', False)
        
        self.input_projection = nn.Linear(config['input_size'], self.d_model)
        
        if self.use_quantum_features:
            self.quantum_feature_projection = nn.Linear(config['quantum_feature_size'], self.d_model)
        
        self.pos_encoder = PositionalEncoding(self.d_model, config['dropout'])
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=config['n_head'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config['num_layers'])
        
        self.decoder = nn.Linear(self.d_model, config['forecast_horizon'])

    def forward(self, x_classical, x_quantum=None):
        classical_proj = self.input_projection(x_classical)
        
        if self.use_quantum_features and x_quantum is not None:
            quantum_proj = self.quantum_feature_projection(x_quantum)
            fused_features = classical_proj + quantum_proj
        else:
            fused_features = classical_proj
        
        fused_features = fused_features * math.sqrt(self.d_model)
        fused_features = self.pos_encoder(fused_features)
        output = self.transformer_encoder(fused_features)
        
        output = output[:, -1, :]
        output = self.decoder(output)
        
        return output