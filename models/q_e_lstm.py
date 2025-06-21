import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np # Use PennyLane's wrapped numpy
from .base_model import BaseModel

# --- Define the Quantum Part of the Model ---
# This section creates the Variational Quantum Circuit (VQC)

# 1. Define the quantum device. We use 'default.qubit', a simulator.
#    The number of qubits is a hyperparameter we will get from the config.
n_qubits = 4 # Default, will be overridden by config
dev = qml.device("default.qubit", wires=n_qubits)

# 2. Define the Quantum Node (The Circuit Itself)
#    This is where the quantum computation happens.
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    The Variational Quantum Circuit (VQC).
    - 'inputs' are the classical features from the LSTM.
    - 'weights' are the trainable parameters of the quantum circuit.
    """
    # a. Encoding Layer: Embed the classical data into the quantum state.
    #    We use AngleEmbedding, which maps each classical feature to a qubit's rotation angle.
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # b. Variational/Training Layer: This is the part of the circuit that "learns".
    #    We use a standard structure of alternating rotation and entanglement layers.
    #    This is similar to a "StronglyEntanglingLayers" ansatz mentioned in many papers.
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # c. Measurement: We measure the expectation value of the Pauli-Z operator for each qubit.
    #    This extracts the classical information back out of the circuit.
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QEnhancedLSTMModel(BaseModel):
    """
    A Hybrid Quantum-Classical LSTM Model.
    It uses a classical LSTM to learn temporal patterns and a quantum circuit
    to enhance feature representation.
    """
    def __init__(self, config):
        super(QEnhancedLSTMModel, self).__init__(config)
        
        # --- Unpack Hyperparameters from Config ---
        # Classical part
        self.input_size = self.config['input_size']
        self.lstm_hidden_size = self.config['classical_lstm_hidden_size']
        self.lstm_num_layers = self.config['num_classical_lstm_layers']
        
        # Quantum part
        # Update the global n_qubits variable for the circuit definition
        global n_qubits
        n_qubits = self.config['n_qubits']
        self.n_qubits = n_qubits
        n_quantum_layers = self.config['n_quantum_layers']
        
        # Other parameters
        dropout = self.config['dropout']
        output_size = self.config['forecast_horizon']

        # --- Define Model Layers ---
        # 1. Classical LSTM Layer
        self.lstm = nn.LSTM(
            self.input_size, 
            self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=dropout if self.lstm_num_layers > 1 else 0
        )

        # 2. Quantum Layer
        # We need to determine the shape of the trainable weights for our quantum circuit.
        weight_shapes = {"weights": (n_quantum_layers, n_qubits, 3)}
        
        # Create the quantum layer using PennyLane's Torch integration.
        # This makes the quantum circuit behave just like a regular PyTorch layer.
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # 3. Final Classical Fully-Connected Layer
        # The input to this layer is the combined output of the classical LSTM and the quantum layer.
        combined_input_size = self.lstm_hidden_size + self.n_qubits
        self.fc = nn.Linear(combined_input_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # 1. Pass data through the classical LSTM
        lstm_out, (hidden_state, _) = self.lstm(x)
        
        # We use the final hidden state of the LSTM as the input for our quantum circuit.
        # This state is a compact representation of the entire input sequence.
        # The hidden state shape is (num_layers, batch_size, lstm_hidden_size).
        # We take the last layer's hidden state.
        classical_features = hidden_state[-1]
        
        # The quantum circuit expects input features to be scaled, often to be within [0, pi].
        # The LSTM's hidden state is already in a reasonable range, but we can add a simple
        # activation like Tanh to ensure the values are bounded between -1 and 1.
        # Then, we can scale them to the range expected by AngleEmbedding.
        quantum_input = torch.tanh(classical_features)
        
        # The number of features for the quantum circuit must match the number of qubits.
        # If lstm_hidden_size is larger than n_qubits, we can use a linear layer to reduce it.
        # For this implementation, we will assume lstm_hidden_size == n_qubits for simplicity.
        # A more robust version would add a `nn.Linear(self.lstm_hidden_size, self.n_qubits)` here.
        if self.lstm_hidden_size != self.n_qubits:
            # For now, we will truncate or pad. A linear layer is a better approach.
            if self.lstm_hidden_size > self.n_qubits:
                quantum_input = quantum_input[:, :self.n_qubits]
            else:
                padding = torch.zeros(quantum_input.shape[0], self.n_qubits - self.lstm_hidden_size, device=x.device)
                quantum_input = torch.cat([quantum_input, padding], dim=1)
        
        # 2. Pass the classical features through the quantum layer
        quantum_features = self.quantum_layer(quantum_input)
        
        # 3. Combine classical and quantum features
        # We concatenate the output of the quantum circuit with the final hidden state of the LSTM.
        combined_features = torch.cat((classical_features, quantum_features), dim=1)
        
        # 4. Pass the combined features through the final fully-connected layer
        out = self.fc(combined_features)
        
        return out
