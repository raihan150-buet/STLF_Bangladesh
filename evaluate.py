import torch
import numpy as np
import pandas as pd
import yaml
import wandb
import os
import matplotlib.pyplot as plt
import argparse

from utils.preprocessing import load_and_preprocess_data
from utils.metrics import calculate_metrics
from utils.plotting import plot_predictions
from utils.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader
from models import get_model

def evaluate(config_path, model_checkpoint_path, data_path):
    """Evaluates a trained model on the test set using the new preprocessing logic."""
    # Load config and inject data path
    with open(config_path, 'r') as f:
        eval_config = yaml.safe_load(f)
    eval_config['data_path'] = data_path

    # Initialize W&B for evaluation
    if eval_config.get("use_wandb", True):
        run_name = f"eval_{eval_config.get('model_type', 'model')}_{os.path.basename(model_checkpoint_path)}"
        wandb.init(
            project=eval_config.get("wandb_project", "default_project"),
            entity=eval_config.get("wandb_entity"),
            config=eval_config,
            name=run_name,
            reinit=True
        )
        print(f"W&B Run for evaluation: {wandb.run.name}")

    # Load data using the enhanced preprocessing function
    # It now returns separate scalers for features and target.
    data_info = load_and_preprocess_data(eval_config)
    target_scaler = data_info["target_scaler"]
    
    test_loader = DataLoader(
        TimeSeriesDataset(data_info["X_test"], data_info["y_test"]),
        batch_size=eval_config["batch_size"],
        shuffle=False
    )

    # Load model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint_path}")
    
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    model = get_model(model_config["model_type"], model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get predictions
    all_preds_scaled, all_actuals_scaled = [], []
    with torch.no_grad():
        for X_batch, y_batch_scaled in test_loader:
            X_batch = X_batch.to(device).float()
            
            preds_scaled = model(X_batch)
            all_preds_scaled.append(preds_scaled.cpu().numpy())
            all_actuals_scaled.append(y_batch_scaled.cpu().numpy())

    final_preds_scaled = np.concatenate(all_preds_scaled)
    final_actuals_scaled = np.concatenate(all_actuals_scaled)

    # --- UPDATED INVERSE TRANSFORM ---
    # Use the dedicated target_scaler for a much cleaner inverse transform.
    final_preds_unscaled = target_scaler.inverse_transform(final_preds_scaled)
    final_actuals_unscaled = target_scaler.inverse_transform(final_actuals_scaled)
    
    # Calculate and print metrics
    metrics = calculate_metrics(final_actuals_unscaled, final_preds_unscaled)
    print("Evaluation Metrics:", metrics)

    # Log to W&B
    if eval_config.get("use_wandb", True) and wandb.run:
        wandb.log(metrics)
        
        # Create and log prediction plot
        fig = plt.figure(figsize=(15, 8)) # Adjusted size for better viewing
        plot_predictions(
            final_actuals_unscaled, 
            final_preds_unscaled, 
            n_samples=5,
            forecast_horizon=eval_config["forecast_horizon"]
        )
        wandb.log({"Test_Set_Predictions_Plot": wandb.Image(fig)})
        plt.close(fig)

        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained forecasting model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    
    args = parser.parse_args()
    evaluate(args.config, args.model, args.data_path)
