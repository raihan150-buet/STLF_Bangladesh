import torch
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from utils.fused_preprocessing import load_and_prepare_fused_data
from utils.metrics import calculate_metrics
from utils.plotting import plot_test_set_results
from utils.data_loader import FusedTimeSeriesDataset
from torch.utils.data import DataLoader
from models import get_model

def evaluate_fused(config_path, model_checkpoint_path, data_path, quantum_feature_path):
    """
    Evaluates a trained fused model (like QuantumResidualTransformer) on the test set.
    """
    print("--- Starting Fused Model Evaluation ---")
    
    # 1. Load Config
    with open(config_path, 'r') as f:
        eval_config = yaml.safe_load(f)
    eval_config.update({
        'data_path': data_path,
        'quantum_feature_path': quantum_feature_path
    })

    # 2. Initialize W&B
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

    # 3. Load Data and Scalers
    # This will return the test sets and the all-important target_scaler
    data_info = load_and_prepare_fused_data(eval_config)
    target_scaler = data_info["target_scaler"]
    
    test_dataset = FusedTimeSeriesDataset(
        data_info['X_classical_test'], 
        data_info['y_test'], 
        data_info.get('X_quantum_test')
    )
    test_loader = DataLoader(test_dataset, batch_size=eval_config.get("batch_size", 32), shuffle=False)

    # 4. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint_path}")
    
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    model = get_model(model_config["model_type"], model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 5. Get Predictions
    all_preds_scaled, all_actuals_scaled = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            if eval_config.get('use_quantum_features', False):
                x_classical, x_quantum, y_batch_scaled = batch
                preds_scaled = model(x_classical.to(device), x_quantum.to(device))
            else:
                x_classical, _, y_batch_scaled = batch
                preds_scaled = model(x_classical.to(device))
            
            all_preds_scaled.append(preds_scaled.cpu().numpy())
            all_actuals_scaled.append(y_batch_scaled.cpu().numpy())

    final_preds_scaled = np.concatenate(all_preds_scaled)
    final_actuals_scaled = np.concatenate(all_actuals_scaled)

    # 6. Inverse Transform to Original Scale
    final_preds_unscaled = target_scaler.inverse_transform(final_preds_scaled)
    final_actuals_unscaled = target_scaler.inverse_transform(final_actuals_scaled)
    
    # 7. Calculate and Log Metrics
    metrics = calculate_metrics(final_actuals_unscaled, final_preds_unscaled)
    print("Evaluation Metrics:", metrics)

    if eval_config.get("use_wandb", True) and wandb.run:
        wandb.log(metrics)
        
        print("Generating evaluation plot...")
        fig = plot_test_set_results(
            y_true_unscaled=final_actuals_unscaled,
            y_pred_unscaled=final_preds_unscaled,
            n_samples=5
        )
        wandb.log({"Test_Set_Evaluation_Plot": wandb.Image(fig)})
        plt.close(fig)

        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fused model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--model", required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--data_path", required=True, help="Path to the original classical data file.")
    parser.add_argument("--quantum_feature_path", default=None, help="Optional path to the pre-computed quantum features file.")
    
    args = parser.parse_args()
    evaluate_fused(args.config, args.model, args.data_path, args.quantum_feature_path)