import torch
import torch.nn as nn
from .base_model import BaseModel

class Classical_Sandwich_LSTM_Cell(nn.Module):
    """
    The purely classical counterpart to the L-QLSTM Cell.
    It replaces the quantum circuit with a standard dense layer and ReLU activation.
    """
    def __init__(self, input_size, hidden_size, enhancer_size):
        super(Classical_Sandwich_LSTM_Cell, self).__init__()
        self.hidden_size = hidden_size

        # First classical layer (same as L-QLSTM)
        self.input_layer = nn.Linear(input_size, 4 * hidden_size)
        self.recurrent_layer = nn.Linear(hidden_size, 4 * hidden_size)

        # --- Classical Enhancer Block (replaces the quantum sandwich) ---
        # This block takes the gate outputs and processes them through a simple feed-forward network.
        self.to_enhancer = nn.Linear(4 * hidden_size, enhancer_size)
        self.relu = nn.ReLU()
        self.from_enhancer = nn.Linear(enhancer_size, 4 * hidden_size)
        # --- End of Classical Block ---

    def forward(self, x, states):
        h_prev, c_prev = states

        # 1. First Classical Layer
        gates_out = self.input_layer(x) + self.recurrent_layer(h_prev)

        # 2. Classical Enhancer "Sandwich"
        enhanced_out = self.to_enhancer(gates_out)
        enhanced_out = self.relu(enhanced_out)
        processed_gates = self.from_enhancer(enhanced_out)
        
        # 3. Standard LSTM gate logic
        i, f, g, o = torch.chunk(processed_gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_next = (f * c_prev) + (i * g)
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class Classical_Sandwich_LSTM(BaseModel):
    """
    The full classical benchmark model that wraps the custom classical cell.
    """
    def __init__(self, config):
        super(Classical_Sandwich_LSTM, self).__init__(config)
        self.hidden_size = config['hidden_size']

        self.lstm_cell = Classical_Sandwich_LSTM__Cell(
            input_size=config['input_size'],
            hidden_size=self.hidden_size,
            # For a fair comparison, the size of this enhancer layer should be
            # comparable to the number of qubits in the L-QLSTM model.
            enhancer_size=config['classical_enhancer_size']
        )
        
        self.fc = nn.Linear(self.hidden_size, config['forecast_horizon'])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t, :], (h, c))
        
        out = self.fc(h)
        return out
