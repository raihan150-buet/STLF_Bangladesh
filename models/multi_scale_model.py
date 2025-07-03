import torch
import torch.nn as nn
from .base_model import BaseModel

class MultiScaleModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        input_size = config['input_size']

        # Sequence length validation
        self.cnn_seq_len = config['cnn_seq_len']
        self.gru_seq_len = config['gru_seq_len']
        self.lstm_seq_len = config['lstm_seq_len']
        assert self.lstm_seq_len >= self.gru_seq_len >= self.cnn_seq_len

        # --- Branch Definitions (driven by config) ---
        cnn_params = config['cnn_params']
        gru_params = config['gru_params']
        lstm_params = config['lstm_params']
        fusion_dim = config['fusion_params']['fusion_dim']

        self.cnn_branch = nn.Sequential(
            nn.Conv1d(input_size, cnn_params['out_channels'],
                      kernel_size=cnn_params['kernel_size'], padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.gru_branch = nn.GRU(input_size, gru_params['hidden_size'],
                                 num_layers=gru_params['num_layers'], batch_first=True)
        self.lstm_branch = nn.LSTM(input_size, lstm_params['hidden_size'],
                                   num_layers=lstm_params['num_layers'], batch_first=True)

        # --- Unified Dimension Projection using nn.ModuleDict ---
        self.projection = nn.ModuleDict({
            'cnn': nn.Linear(cnn_params['out_channels'], fusion_dim),
            'gru': nn.Linear(gru_params['hidden_size'], fusion_dim),
            'lstm': nn.Linear(lstm_params['hidden_size'], fusion_dim)
        })

        # --- Attention Fusion ---
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True # Important for correct shape handling
        )
        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, config['forecast_horizon'])
        )

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        
        # CNN Branch
        cnn_out = self.cnn_branch(x[:, -self.cnn_seq_len:, :].permute(0, 2, 1))
        cnn_proj = self.projection['cnn'](cnn_out)

        # GRU Branch
        gru_full_out, _ = self.gru_branch(x[:, -self.gru_seq_len:, :])
        gru_proj = self.projection['gru'](gru_full_out[:, -1, :])

        # LSTM Branch
        lstm_full_out, _ = self.lstm_branch(x)
        lstm_proj = self.projection['lstm'](lstm_full_out[:, -1, :])

        # Stack and fuse with attention
        combined = torch.stack([cnn_proj, gru_proj, lstm_proj], dim=1)
        attn_out, _ = self.attention(combined, combined, combined)
        
        # Aggregate and produce final prediction
        fusion_input = attn_out.mean(dim=1)
        return self.fusion_layer(fusion_input).squeeze(-1)