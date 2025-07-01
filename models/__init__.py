from .base_model import BaseModel

# --- Import Classical Models ---
from .lstm import LSTMModel
from .tcn import TCNModel
from .cnn_lstm import CNN_LSTM_Model
from .transformer_model import TransformerModel

# --- Import Quantum-Inspired and SOTA Classical Models ---
from .quantum_inspired_lstm import QuantumInspiredLSTM
from .classical_conv_lstm import ClassicalConvLSTM

# --- Import Hybrid Quantum & Benchmark Models ---
from .q_e_lstm import QEnhancedLSTMModel
from .classical_enhanced_lstm import ClassicalEnhancedLSTMModel
from .qdi_lstm import QDILSTMModel
from .classical_qdi_benchmark import ClassicalQDIBenchmarkModel
from .qres_lstm import QResLSTM
from .classical_res_lstm import ClassicalResLSTM

from .quantum_residual_transformer import QuantumResidualTransformer


# A dictionary mapping the model_type string from your config to the actual model class.

MODEL_REGISTRY = {
    # Standard Classical
    "lstm": LSTMModel,
    "tcn": TCNModel,
    "cnn_lstm": CNN_LSTM_Model,
    "transformer": TransformerModel,
    
    # Advanced Classical & Quantum-Inspired
    "quantum_inspired_lstm": QuantumInspiredLSTM,
    "classical_conv_lstm": ClassicalConvLSTM,
    
    # Hybrid Quantum Models
    "qenhancedlstm": QEnhancedLSTMModel,
    "qdi_lstm": QDILSTMModel,
    "qres_lstm": QResLSTM,
    
    # Corresponding Classical Benchmarks
    "classical_enhanced_lstm": ClassicalEnhancedLSTMModel,
    "classical_qdi_benchmark": ClassicalQDIBenchmarkModel,
    "classical_res_lstm": ClassicalResLSTM,

    # Fused Feature Model
    "quantum_residual_transformer": QuantumResidualTransformer
}

def get_model(model_type: str, config: dict) -> BaseModel:
    """
    Factory function to return a model instance based on its type.
    
    Args:
        model_type (str): The name of the model, as defined in the config file.
        config (dict): The configuration dictionary for the experiment.
        
    Returns:
        An instance of the requested model class.
    """
    model_type_lower = model_type.lower()
    
    model_class = MODEL_REGISTRY.get(model_type_lower)
    
    if model_class:
        # All our models are designed to accept the entire config dictionary
        return model_class(config)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Please check your config file and models/__init__.py.")