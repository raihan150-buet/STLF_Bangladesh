import torch
import torch.nn as nn
import pennylane as qml
from .base_model import BaseModel

# Define the quantum components globally, configured by the model's __init__
n_qubits = 4 # Default value
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def vqc_for_lstm_cell(inputs, weights):
    """A standard VQC to be used inside the LSTM cell."""
    qml.AngleEmbedding(inputs, wires=range(dev.num_wires))
    qml.BasicEntanglerLayers(weights, wires=range(dev.num_wires))
    return [qml.expval(qml.PauliZ(i)) for i in range(dev.num_wires)]

class L_QLSTM_Cell(nn.Module):
    """
    A single Quantum-Enhanced LSTM Cell, implementing the "sandwich" architecture.
    """
    def __init__(self, input_size, hidden_size, n_qubits, n_quantum_layers):
        super(L_QLSTM_Cell, self).__init__()
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits

        # Classical input and recurrent layers (maps input and hidden state to the 4 gates)
        self.input_layer = nn.Linear(input_size, 4 * hidden_size)
        self.recurrent_layer = nn.Linear(hidden_size, 4 * hidden_size)

        # Quantum layer (sandwiched in the middle)
        # It takes the output of the classical gates and processes it.
        # Its input size must be n_qubits.
        self.to_quantum = nn.Linear(4 * hidden_size, n_qubits)
        weight_shapes = {"weights": (n_quantum_layers, n_qubits)}
        self.quantum_layer = qml.qnn.TorchLayer(vqc_for_lstm_cell, weight_shapes)
        
        # Final classical layer after the quantum circuit
        self.final_classical_layer = nn.Linear(n_qubits, 4 * hidden_size)

    def forward(self, x, states):
        # x shape: (batch_size, input_size)
        h_prev, c_prev = states
        # h_prev, c_prev shapes: (batch_size, hidden_size)

        # 1. First Classical Layer
        gates_out = self.input_layer(x) + self.recurrent_layer(h_prev)

        # 2. Quantum Layer (The "Sandwich")
        quantum_in = self.to_quantum(gates_out)
        quantum_out = self.quantum_layer(quantum_in)

        # 3. Second Classical Layer
        processed_gates = self.final_classical_layer(quantum_out)
        
        # Split the result into the 4 LSTM gates
        i, f, g, o = torch.chunk(processed_gates, 4, dim=1)
        
        # Standard LSTM operations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_next = (f * c_prev) + (i * g)
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class L_QLSTM(BaseModel):
    """
    The full L-QLSTM model that wraps the custom cell and runs it over a sequence.
    """
    def __init__(self, config):
        super(L_QLSTM, self).__init__(config)

        # Configure the quantum device
        global dev
        self.n_qubits = config['n_qubits']
        dev.wires = qml.wires.Wires(range(self.n_qubits))

        self.hidden_size = config['hidden_size']

        self.lstm_cell = L_QLSTM_Cell(
            input_size=config['input_size'],
            hidden_size=self.hidden_size,
            n_qubits=self.n_qubits,
            n_quantum_layers=config['n_quantum_layers']
        )
        
        self.fc = nn.Linear(self.hidden_size, config['forecast_horizon'])

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Loop over the time steps
        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t, :], (h, c))
        
        # The output of the final time step is used for prediction
        out = self.fc(h)
        return out
