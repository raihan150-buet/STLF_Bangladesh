import torch
import torch.nn as nn
import pennylane as qml
from .base_model import BaseModel

class QDILSTM(BaseModel):
    """A dedicated implementation of the HQRNN-inspired model, now with instance-specific device."""
    def __init__(self, config):
        super(QDILSTM, self).__init__(config)
        
        self.n_qubits = self.config['n_qubits']
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def qdi_circuit(inputs, weights_v1, weights_v2):
            for i in range(self.n_qubits):
                qml.RX(weights_v1[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(self.n_qubits):
                qml.RZ(inputs[i], wires=i)
            for i in range(self.n_qubits):
                qml.RX(weights_v2[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)]
        
        self.lstm = nn.LSTM(
            self.config['input_size'], 
            self.config['classical_lstm_hidden_size'],
            num_layers=self.config['num_classical_lstm_layers'],
            batch_first=True,
            dropout=self.config['dropout'] if self.config['num_classical_lstm_layers'] > 1 else 0
        )
        
        self.to_quantum = nn.Linear(self.config['classical_lstm_hidden_size'], self.n_qubits)
        
        weight_shapes = {
            "weights_v1": (self.n_qubits,),
            "weights_v2": (self.n_qubits,)
        }
        self.quantum_layer = qml.qnn.TorchLayer(qdi_circuit, weight_shapes)

        combined_input_size = self.config['classical_lstm_hidden_size'] + self.n_qubits
        self.fc = nn.Linear(combined_input_size, self.config['forecast_horizon'])

    def forward(self, x):
        _, (hidden_state, _) = self.lstm(x)
        classical_features = hidden_state[-1]
        
        quantum_input = self.to_quantum(classical_features)
        quantum_features = self.quantum_layer(quantum_input)
        
        combined_features = torch.cat((classical_features, quantum_features), dim=1)
        out = self.fc(combined_features)
        return out
