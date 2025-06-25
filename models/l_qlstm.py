import torch
import torch.nn as nn
import pennylane as qml
from .base_model import BaseModel

class L_QLSTM_Cell(nn.Module):
    """
    A single Quantum-Enhanced LSTM Cell, implementing the "sandwich" architecture.
    """
    def __init__(self, input_size, hidden_size, n_qubits, n_quantum_layers, quantum_device):
        super(L_QLSTM_Cell, self).__init__()
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits

        # --- Define the Quantum Node inside the Cell ---
        # It uses the device passed from the main model.
        @qml.qnode(quantum_device, interface="torch")
        def vqc_for_lstm_cell(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Classical layers
        self.input_layer = nn.Linear(input_size, 4 * hidden_size)
        self.recurrent_layer = nn.Linear(hidden_size, 4 * hidden_size)
        self.to_quantum = nn.Linear(4 * hidden_size, n_qubits)
        
        # Quantum layer
        weight_shapes = {"weights": (n_quantum_layers, n_qubits)}
        self.quantum_layer = qml.qnn.TorchLayer(vqc_for_lstm_cell, weight_shapes)
        
        # Final classical layer
        self.final_classical_layer = nn.Linear(n_qubits, 4 * hidden_size)

    def forward(self, x, states):
        h_prev, c_prev = states
        gates_out = self.input_layer(x) + self.recurrent_layer(h_prev)
        quantum_in = self.to_quantum(gates_out)
        quantum_out = self.quantum_layer(quantum_in)
        processed_gates = self.final_classical_layer(quantum_out)
        i, f, g, o = torch.chunk(processed_gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = (f * c_prev) + (i * g)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class L_QLSTM(BaseModel):
    """The full L-QLSTM model that wraps the custom cell."""
    def __init__(self, config):
        super(L_QLSTM, self).__init__(config)

        # --- Create an instance-specific quantum device ---
        self.n_qubits = config['n_qubits']
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        self.hidden_size = config['hidden_size']

        # Pass the newly created device to the cell
        self.lstm_cell = L_QLSTM_Cell(
            input_size=config['input_size'],
            hidden_size=self.hidden_size,
            n_qubits=self.n_qubits,
            n_quantum_layers=config['n_quantum_layers'],
            quantum_device=self.dev
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
