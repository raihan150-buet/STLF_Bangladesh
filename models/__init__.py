# This file makes the 'models' directory a Python package.
# It also contains the factory function to get a model by name.

from .lstm import LSTMModel
from .tcn import TCNModel
from .q_e_lstm import QEnhancedLSTMModel
from .base_model import BaseModel
from .transformer import TransformerModel
from .cnn_lstm import CNN_LSTM_Model
from .classical_enhanced_lstm import ClassicalEnhancedLSTMModel
from .quantum_inspired_lstm import QuantumInspiredLSTM
from .classical_conv_lstm import ClassicalConvLSTM
from .qdi_lstm import QDILSTM
from .classical_qdi import ClassicalQDI
from .qres_lstm import QResLSTM
from .classical_res_lstm import ClassicalResLSTM


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
    elif model_type == "cnn_lstm":
        return CNN_LSTM_Model(config)
    elif model_type == "classical_enhanced_lstm":
        return ClassicalEnhancedLSTMModel(config)
    elif model_type == "quantum_inspired_lstm":
        return QuantumInspiredLSTM(config)
    elif model_type == "classical_conv_lstm":
        return ClassicalConvLSTM(config)
    elif model_type == "qdi_lstm":
        return QDILSTM(config)
    elif model_type == "classical_qdi":
        return ClassicalQDI(config)
    elif model_type == "qres_lstm":
        return QResLSTM(config)
    elif model_type == "classical_res_lstm":
        return ClassicalResLSTM(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

