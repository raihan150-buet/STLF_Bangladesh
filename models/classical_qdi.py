import torch
import torch.nn as nn
from .base_model import BaseModel

class ClassicalQDI(BaseModel):
    """
    An 'architecturally equivalent' classical counterpart to the QDI_LSTM model.
    
    This model replaces the quantum circuit with a simple dense layer (a small
    feed-forward network) to act as a classical 'enhancer'.
    """
    def __init__(self, config):
        super(ClassicalQDI, self).__init__(config)
        
        # --- Unpack Hyperparameters ---
        # Classical LSTM part (identical to the QDI-LSTM)
        self.lstm_hidden_size = self.config['classical_lstm_hidden_size']
        
        # Classical Enhancer Part
        # This size should match 'n_qubits' in the quantum model for a fair comparison
        self.enhancer_output_size = self.config['classical_enhancer_size']
        
        # --- Define Model Layers ---
        # 1. Classical LSTM Layer (Identical to the QDI-LSTM model)
        self.lstm = nn.LSTM(
            input_size=self.config['input_size'], 
            hidden_size=self.lstm_hidden_size,
            num_layers=self.config['num_classical_lstm_layers'],
            batch_first=True,
            dropout=self.config['dropout'] if self.config['num_classical_lstm_layers'] > 1 else 0
        )

        # 2. Classical Enhancer Layer (This replaces the quantum circuit)
        # It's a simple feed-forward block to process the LSTM's output.
        self.classical_enhancer = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.enhancer_output_size),
            nn.ReLU()
        )

        # 3. Final Fully-Connected Layer
        # The input size is the sum of the LSTM output and the enhancer output.
        combined_input_size = self.lstm_hidden_size + self.enhancer_output_size
        self.fc = nn.Linear(combined_input_size, self.config['forecast_horizon'])

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # 1. Pass data through the classical LSTM
        _, (hidden_state, _) = self.lstm(x)
        
        # Use the final hidden state as the input for the enhancer
        classical_features = hidden_state[-1]
        
        # 2. Pass the features through the classical enhancer layer
        enhanced_features = self.classical_enhancer(classical_features)
        
        # 3. Combine original and enhanced features
        combined_features = torch.cat((classical_features, enhanced_features), dim=1)
        
        # 4. Pass through the final fully-connected layer
        out = self.fc(combined_features)
        
        return out
