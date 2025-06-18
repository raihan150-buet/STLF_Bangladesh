# This file makes the 'models' directory a Python package.
# It also contains the factory function to get a model by name.

from .lstm import LSTMModel
from .tcn import TCNModel
from .q_e_lstm import QEnhancedLSTMModel
from .base_model import BaseModel
from .transformer import TransformerModel


def get_model(model_type, config):
    """
    Factory function to return a model instance based on its type.
    """
    model_type = model_type.lower()
    
    # CORRECTED: All models that are defined to take a 'config' object
    # should be called the same way for consistency.
    
    if model_type == "lstm":
        return LSTMModel(config)
    elif model_type == "tcn":
        return TCNModel(config)
    elif model_type == "qenhancedlstm":
        return QEnhancedLSTMModel(config)
    elif model_type == "transformer":
        return TransformerModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

