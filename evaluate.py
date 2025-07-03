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
# Import the new plotting function
from utils.plotting import plot_test_set_results
from utils.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader
from models import get_model

def evaluate(config_path, model_checkpoint_path, data_path):
    """Evaluates a trained model on the test set."""
    with open(config_path, 'r') as f:
        eval_config = yaml.safe_load(f)
    eval_config['data_path'] = data_path

    if eval_config.get("use_wandb", True):
        run_name = f"eval_{os.path.basename(model_checkpoint_path).replace('.pth', '')}"
        wandb.init(
            project=eval_config.get("wandb_project", "default_project"),
            entity=eval_config.get("wandb_entity"),
            config=eval_config,
            name=run_name,
            reinit=True
        )
        print(f"W&B Run for evaluation: {wandb.run.name}")

    data_info = load_and_preprocess_data(eval_config)
    target_scaler = data_info["target_scaler"]
    
    test_loader = DataLoader(
        TimeSeriesDataset(data_info["X_test"], data_info["y_test"]),
        batch_size=eval_config.get("batch_size", 64),
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint_path}")
    
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['config']
    model = get_model(model_config["model_type"], model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds_scaled, all_actuals_scaled = [], []
    with torch.no_grad():
        for X_batch, y_batch_scaled in test_loader:
            preds_scaled = model(X_batch.to(device))
            all_preds_scaled.append(preds_scaled.cpu().numpy())
            all_actuals_scaled.append(y_batch_scaled.cpu().numpy())

    final_preds_scaled = np.concatenate(all_preds_scaled)
    final_actuals_scaled = np.concatenate(all_actuals_scaled)

    final_preds_unscaled = target_scaler.inverse_transform(final_preds_scaled)
    final_actuals_unscaled = target_scaler.inverse_transform(final_actuals_scaled)
    
    metrics = calculate_metrics(final_actuals_unscaled, final_preds_unscaled)
    print("Evaluation Metrics:", metrics)

    if eval_config.get("use_wandb", True) and wandb.run:
        wandb.log(metrics)
        
        print("Generating evaluation plot...")
        # Call the new, comprehensive plotting function
        fig = plot_test_set_results(
            y_true_unscaled=final_actuals_unscaled,
            y_pred_unscaled=final_preds_unscaled,
            n_samples=5 # How many full horizon samples to overlay
        )
        
        # Log the generated figure to Weights & Biases
        wandb.log({"Test_Set_Evaluation_Plot": wandb.Image(fig)})
        plt.close(fig) # Close the figure to free up memory

        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained forecasting model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    
    args = parser.parse_args()
    evaluate(args.config, args.model, args.data_path)
