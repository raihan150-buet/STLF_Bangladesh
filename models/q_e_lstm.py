# models/q_e_lstm.py
import torch
import torch.nn as nn
import pennylane as qml # Import PennyLane here
from .base_model import BaseModel

# --- Definition of QuantumLayer (moved here) ---
class QuantumLayer(nn.Module):
    def __init__(self, n_input_features_for_q_layer, n_qubits, n_pqc_layers, 
                 output_dim=None, q_device_name="default.qubit", 
                 diff_method="backprop",
                 ansatz_type="StronglyEntanglingLayers"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_pqc_layers = n_pqc_layers 
        self.q_output_dim = output_dim if output_dim is not None else self.n_qubits
        self.dev = qml.device(q_device_name, wires=self.n_qubits)
        self.ansatz_type = ansatz_type
        self.n_features_for_embedding = n_qubits 
        self.diff_method_for_qnode = diff_method

        if n_input_features_for_q_layer < self.n_features_for_embedding:
            print(f"Warning: QuantumLayer received n_input_features_for_q_layer ({n_input_features_for_q_layer}) "
                  f"< n_features_for_embedding ({self.n_features_for_embedding}).")
        elif n_input_features_for_q_layer > self.n_features_for_embedding:
             print(f"Warning: QuantumLayer received n_input_features_for_q_layer ({n_input_features_for_q_layer}) "
                  f"> n_features_for_embedding ({self.n_features_for_embedding}). "
                  "AngleEmbedding will use the first n_features_for_embedding features.")

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def _quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_features_for_embedding), rotation='Y')

            if self.ansatz_type == "StronglyEntanglingLayers":
                qml.StronglyEntanglingLayers(weights=weights, wires=range(self.n_qubits))
            elif self.ansatz_type == "BasicEntanglerLayers":
                 qml.BasicEntanglerLayers(weights=weights, wires=range(self.n_qubits), rotation=qml.RY)
            else: 
                for layer_idx in range(self.n_pqc_layers):
                    for i in range(self.n_qubits):
                        qml.Rot(weights[layer_idx, i, 0],
                                weights[layer_idx, i, 1],
                                weights[layer_idx, i, 2], wires=i)
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if self.n_qubits > 1 and self.n_pqc_layers > 0:
                        qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.q_output_dim)]

        if self.ansatz_type == "StronglyEntanglingLayers":
            weight_shapes = {"weights": (self.n_pqc_layers, self.n_qubits, 3)}
        elif self.ansatz_type == "BasicEntanglerLayers":
            weight_shapes = {"weights": (self.n_pqc_layers, self.n_qubits)}
        else: 
            weight_shapes = {"weights": (self.n_pqc_layers, self.n_qubits, 3)}
            
        self.qnn_layer = qml.qnn.TorchLayer(_quantum_circuit, weight_shapes)

    def forward(self, classical_inputs):
        if classical_inputs.shape[-1] < self.n_features_for_embedding:
            raise ValueError(f"QuantumLayer expects at least {self.n_features_for_embedding} input features, "
                             f"got {classical_inputs.shape[-1]}.")
        
        inputs_for_qml = classical_inputs[:, :self.n_features_for_embedding]
        return self.qnn_layer(inputs_for_qml)
# --- End of QuantumLayer Definition ---


class QEnhancedLSTMModel(BaseModel): # Your QEnhancedLSTMModel definition remains the same
    def __init__(self, config):
        super().__init__(config)
        
        self.input_size = config["input_size"]
        self.classical_lstm_hidden_size = config.get("classical_lstm_hidden_size", config.get("hidden_size", 64))
        self.num_classical_lstm_layers = config.get("num_classical_lstm_layers", config.get("num_layers", 1))
        self.classical_dropout = config.get("dropout", 0.1) if self.num_classical_lstm_layers > 1 else 0
        
        self.forecast_horizon = config["forecast_horizon"]

        self.n_qubits_qelstm = config.get("n_qubits_qelstm", 4) 
        self.n_pqc_layers_qelstm = config.get("n_pqc_layers_qelstm", 2) 
        self.ansatz_type_qelstm = config.get("ansatz_type_qelstm", "StronglyEntanglingLayers")

        self.lstm_layer = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.classical_lstm_hidden_size,
            num_layers=self.num_classical_lstm_layers,
            dropout=self.classical_dropout,
            batch_first=True
        )
        
        if self.classical_lstm_hidden_size < self.n_qubits_qelstm:
            raise ValueError(
                f"Classical LSTM hidden size ({self.classical_lstm_hidden_size}) "
                f"must be >= n_qubits_qelstm ({self.n_qubits_qelstm}) for slicing."
            )

        self.quantum_enhancer = QuantumLayer( # Uses the QuantumLayer defined above in this file
            n_input_features_for_q_layer=self.n_qubits_qelstm,
            n_qubits=self.n_qubits_qelstm,
            n_pqc_layers=self.n_pqc_layers_qelstm,
            output_dim=self.n_qubits_qelstm,
            ansatz_type=self.ansatz_type_qelstm
        )
        
        combined_features_dim = self.classical_lstm_hidden_size + self.n_qubits_qelstm
        fc_hidden_dim1 = config.get("qelstm_fc_hidden1_dim", combined_features_dim // 2)
        fc_hidden_dim2 = config.get("qelstm_fc_hidden2_dim", fc_hidden_dim1 // 2)
        fc_dropout = config.get("qelstm_fc_dropout", config.get("dropout", 0.1))

        layers = []
        layers.append(nn.Linear(combined_features_dim, fc_hidden_dim1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(fc_dropout))
        
        if fc_hidden_dim2 > 0 and fc_hidden_dim2 < fc_hidden_dim1 :
            layers.append(nn.Linear(fc_hidden_dim1, fc_hidden_dim2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(fc_dropout))
            layers.append(nn.Linear(fc_hidden_dim2, self.forecast_horizon))
        else:
            layers.append(nn.Linear(fc_hidden_dim1, self.forecast_horizon))

        self.fc_output_layers = nn.Sequential(*layers)
        
    def _scale_input_for_quantum_enhancer(self, x):
        # Example scaling: scale to approx [0, 2*pi] for AngleEmbedding
        return 2 * torch.pi * torch.sigmoid(x) 

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm_layer(x)
        classical_lstm_final_hidden = h_n[-1, :, :]
        q_input_classical_slice = classical_lstm_final_hidden[:, :self.n_qubits_qelstm]
        q_input_scaled = self._scale_input_for_quantum_enhancer(q_input_classical_slice)
        quantum_enhancement_output = self.quantum_enhancer(q_input_scaled)
        combined_representation = torch.cat((classical_lstm_final_hidden, quantum_enhancement_output), dim=1)
        predictions = self.fc_output_layers(combined_representation)
        return predictions