import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """Standard PyTorch Dataset for time series data."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- UPDATED DATASET FOR FUSED FEATURES ---
class FusedTimeSeriesDataset(Dataset):
    """A flexible dataset that can yield classical features only, or both classical and quantum features."""
    def __init__(self, X_classical, y, X_quantum=None):
        self.X_classical = torch.from_numpy(X_classical).float()
        self.y = torch.from_numpy(y).float()
        
        self.has_quantum = X_quantum is not None
        if self.has_quantum:
            self.X_quantum = torch.from_numpy(X_quantum).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.has_quantum:
            return self.X_classical[idx], self.X_quantum[idx], self.y[idx]
        else:
            # Return classical features, a placeholder for quantum, and the target
            return self.X_classical[idx], torch.empty(0), self.y[idx]