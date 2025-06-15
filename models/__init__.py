# This file makes the 'models' directory a Python package.
# It also contains the factory function to get a model by name.

from .lstm import LSTMModel
from .tcn import TCNModel
# from .transformer import TransformerModel # Uncomment when implemented
from .q_e_lstm import QEnhancedLSTMModel

def get_model(model_type, config):
    """
    Factory function to return a model instance based on its type.
    """
    model_type = model_type.lower()
    
    if model_type == "lstm":
        return LSTMModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['forecast_horizon'],
            dropout=config['dropout']
        )
    elif model_type == "tcn":
        return TCNModel(
            input_size=config['input_size'],
            output_size=config['forecast_horizon'],
            num_channels=[config['hidden_size']] * config['num_layers'], # Example channel setup
            kernel_size=3, # Example kernel size, should be in config
            dropout=config['dropout']
        )
    # elif model_type == "transformer":
    #     return TransformerModel(...)
    elif model_type == "qenhancedlstm":
        return QEnhancedLSTMModel(config) # Pass the whole config for complex models
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

