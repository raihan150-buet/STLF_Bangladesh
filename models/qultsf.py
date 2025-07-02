import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from .base_model import BaseModel


class Hybrid_QML_Model(nn.Module):
    def __init__(self, lookback_window_size, forecast_window_size, num_qubits, QML_device, num_layers):
        super(Hybrid_QML_Model, self).__init__()
        self.forecast_window_size = forecast_window_size
        self.dev = qml.device(QML_device, wires=num_qubits)

        @qml.qnode(self.dev, interface='torch', diff_method="best")
        def quantum_function(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(num_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self._quantum_circuit = quantum_function
        q_weights_shape = {'weights': (num_layers, num_qubits, 3)}
        self.input_classical_layer = torch.nn.Linear(lookback_window_size, 2 ** num_qubits)
        self.hidden_quantum_layer = qml.qnn.TorchLayer(self._quantum_circuit, q_weights_shape)
        self.output_classical_layer = torch.nn.Linear(num_qubits, forecast_window_size)

    def forward(self, batch_input):
        y = batch_input.reshape(batch_input.shape[0] * batch_input.shape[1], batch_input.shape[2])
        y = self.input_classical_layer(y)
        y = self.hidden_quantum_layer(y)
        y = self.output_classical_layer(y)
        batch_output = y.reshape(batch_input.shape[0], batch_input.shape[1], self.forecast_window_size)
        return batch_output


class QuLTSF(BaseModel):
    """
    Implementation of the QuLTSF model from arXiv:2412.13769v2 [cite: 2]
    """
    def __init__(self, config):
        super(QuLTSF, self).__init__(config)
        self.seq_len = config['sequence_length']
        self.pred_len = config['forecast_horizon']
        self.num_qubits = config['n_qubits']
        self.QML_device = config.get('qml_device', 'default.qubit')
        self.num_layers = config['n_quantum_layers']

        self.hybrid_qml_model = Hybrid_QML_Model(
            self.seq_len, self.pred_len, self.num_qubits, self.QML_device, self.num_layers
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.hybrid_qml_model(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x  # [Batch, Output length, Channel]