import torch
import torch.nn as nn
import pennylane as qml
from .base_model import BaseModel

class QResLSTM(BaseModel):
    """
    A Quantum-Enhanced LSTM with a Residual Connection.
    This version includes the fix for the PennyLane batching error.
    """
    def __init__(self, config):
        super(QResLSTM, self).__init__(config)
        
        # --- Quantum Configuration ---
        self.n_qubits = config.get('n_qubits', 4)
        self.q_depth = config.get('q_depth', 2)
        self.q_delta = config.get('q_delta', 0.01) # Controls quantum contribution
        
        # Quantum device setup
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Define the Quantum Circuit within the model's scope
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            for i in range(self.n_qubits):
                qml.RY(inputs[i] * torch.pi, wires=i)
            
            for layer in range(self.q_depth):
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CZ(wires=[i, i+1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # --- Classical LSTM Component ---
        self.lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config.get('dropout', 0.1) if config['num_layers'] > 1 else 0
        )
        
        # --- Hybrid Components ---
        weight_shapes = {"weights": (self.q_depth, self.n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        self.to_quantum = nn.Sequential(
            nn.Linear(config['hidden_size'], self.n_qubits),
            nn.Tanh()
        )
        
        self.residual_processor = nn.Sequential(
            nn.Linear(self.n_qubits, config['hidden_size']),
            nn.LayerNorm(config['hidden_size'])
        )
        
        self.fc = nn.Linear(config['hidden_size'], config['forecast_horizon'])

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        quantum_input = self.to_quantum(last_hidden)
        
        # --- CORRECTED BATCH PROCESSING ---
        # Iterate through the batch, process each item individually, then stack.
        quantum_features_list = [self.qlayer(inp) for inp in quantum_input]
        quantum_features = torch.stack(quantum_features_list)
        # --- End of Fix ---
        
        # Use no_grad for stability as per your research
        with torch.no_grad():
            residual = self.residual_processor(quantum_features)
            residual = residual * self.q_delta
        
        hybrid_features = last_hidden + residual
        
        output = self.fc(hybrid_features)
        return output
