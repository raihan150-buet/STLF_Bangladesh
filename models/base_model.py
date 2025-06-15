import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Abstract base class for all forecasting models"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Prediction tensor of shape (batch_size, forecast_horizon)
        """
        pass
    
    def save(self, path):
        """Save model to file
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path, map_location=None):
        """Load model from file
        
        Args:
            path: Path to saved model
            map_location: Device to load model onto
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_parameter_count(self):
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)