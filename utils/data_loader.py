import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    def __init__(self, X, y):
        # Convert to torch tensors
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
