import torch
import torch.nn as nn
from .base_model import BaseModel

class ClassicalEnhancedLSTMModel(BaseModel):
    """
    An 'architecturally equivalent' classical counterpart to the QEnhancedLSTMModel.
    
    This model replaces the quantum circuit with a simple dense layer (a small
    feed-forward network) to act as a classical 'enhancer'.
    """
    def __init__(self, config):
        super(ClassicalEnhancedLSTMModel, self).__init__(config)
        
        # --- Unpack Hyperparameters ---
        # Classical LSTM part (same as the quantum model)
        self.input_size = self.config['input_size']
        self.lstm_hidden_size = self.config['classical_lstm_hidden_size']
        self.lstm_num_layers = self.config['num_classical_lstm_layers']
        
        # Classical Enhancer Part
        # This size should be comparable to n_qubits in the quantum model for a fair test.
        self.enhancer_output_size = self.config['classical_enhancer_size']
        
        # Other parameters
        dropout = self.config['dropout']
        output_size = self.config['forecast_horizon']

        # --- Define Model Layers ---
        # 1. Classical LSTM Layer (Identical to the Q-Enhanced model)
        self.lstm = nn.LSTM(
            self.input_size, 
            self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=dropout if self.lstm_num_layers > 1 else 0
        )

        # 2. Classical Enhancer Layer
        # This replaces the quantum circuit. It's a simple feed-forward block.
        self.classical_enhancer = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.enhancer_output_size),
            nn.ReLU()
        )

        # 3. Final Fully-Connected Layer
        # The input size is the sum of the LSTM output and the enhancer output.
        combined_input_size = self.lstm_hidden_size + self.enhancer_output_size
        self.fc = nn.Linear(combined_input_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # 1. Pass data through the classical LSTM
        lstm_out, (hidden_state, _) = self.lstm(x)
        
        # Use the final hidden state as the input for the enhancer
        classical_features = hidden_state[-1]
        
        # 2. Pass the features through the classical enhancer layer
        enhanced_features = self.classical_enhancer(classical_features)
        
        # 3. Combine original and enhanced features
        combined_features = torch.cat((classical_features, enhanced_features), dim=1)
        
        # 4. Pass through the final fully-connected layer
        out = self.fc(combined_features)
        
        return out
