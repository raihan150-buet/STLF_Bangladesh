import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE, avoiding division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Add a small epsilon to the denominator to avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def calculate_metrics(y_true, y_pred):
    """Calculates and returns a dictionary of common regression metrics."""
    # Ensure y_true and y_pred are flattened for metric calculations
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    return {
        'MAE': mean_absolute_error(y_true_flat, y_pred_flat),
        'MAPE': mean_absolute_percentage_error(y_true_flat, y_pred_flat),
        'RMSE': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        'R2': r2_score(y_true_flat, y_pred_flat)
    }
