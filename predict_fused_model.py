import torch
import numpy as np
import pandas as pd
import yaml
import os
from datetime import timedelta
import argparse
import wandb

# We need to import the preprocessing function to get access to the scalers
from utils.fused_preprocessing import load_and_prepare_fused_data
from models import get_model

def predict_fused(config_path, model_checkpoint_path, data_path, quantum_feature_path, input_csv_path):
    """
    Makes a new forecast using a trained fused model (like QuantumResidualTransformer).
    """
    print("--- Starting Fused Model Prediction ---")
    
    # 1. Load Config
    with open(config_path, 'r') as f:
        pred_config = yaml.safe_load(f)
    pred_config.update({
        'data_path': data_path,
        'quantum_feature_path': quantum_feature_path
    })

    # 2. Load Scalers by doing a "dry run" of the preprocessing
    # This is a crucial step to ensure we use the exact same scaling as in training.
    print("Loading scalers from the training data...")
    data_info = load_and_prepare_fused_data(pred_config)
    classical_feature_scaler = data_info["classical_feature_scaler"]
    target_scaler = data_info["target_scaler"]
    if pred_config.get('use_quantum_features', False):
        quantum_feature_scaler = data_info["quantum_feature_scaler"]

    # 3. Prepare the Input Sequence for Prediction
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV for prediction not found at {input_csv_path}")
    
    input_df = pd.read_excel(input_csv_path, engine='openpyxl')
    input_df['Datetime'] = pd.to_datetime(input_df['Datetime'])
    input_df = input_df.set_index('Datetime').sort_index()

    sequence_length = pred_config["sequence_length"]
    if len(input_df) < sequence_length:
        raise ValueError(f"Input data must have at least {sequence_length} rows.")

    # Take the most recent data needed for one sequence
    df_for_pred = input_df.iloc[-sequence_length:].copy()
    last_known_datetime = df_for_pred.index[-1]

    # Apply the same feature engineering
    df_for_pred['is_weekend'] = df_for_pred.index.dayofweek.isin([4, 5]).astype(int)
    
    # Select the classical features and scale them
    classical_features_to_use = pred_config.get("classical_features", [])
    X_classical_unscaled = df_for_pred[classical_features_to_use].values
    X_classical_scaled = classical_feature_scaler.transform(X_classical_unscaled)
    
    X_classical_tensor = torch.FloatTensor(X_classical_scaled).unsqueeze(0) # Add batch dimension

    # Conditionally load and scale quantum features
    X_quantum_tensor = None
    if pred_config.get('use_quantum_features', False):
        df_quantum_full = pd.read_csv(pred_config["quantum_feature_path"], index_col='Datetime', parse_dates=True)
        # Align the quantum features with the input data
        df_quantum_pred = df_quantum_full.loc[df_for_pred.index]
        quantum_features_to_use = [col for col in df_quantum_pred.columns if 'q_feat' in col]
        X_quantum_unscaled = df_quantum_pred[quantum_features_to_use].values
        X_quantum_scaled = quantum_feature_scaler.transform(X_quantum_unscaled)
        X_quantum_tensor = torch.FloatTensor(X_quantum_scaled).unsqueeze(0)

    # 4. Load Model and Make Prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    model = get_model(model_config["model_type"], model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        if pred_config.get('use_quantum_features', False):
            scaled_prediction = model(X_classical_tensor.to(device), X_quantum_tensor.to(device))
        else:
            scaled_prediction = model(X_classical_tensor.to(device))

    # 5. Inverse Transform the Prediction
    unscaled_prediction = target_scaler.inverse_transform(scaled_prediction.cpu().numpy())

    # 6. Create Results DataFrame
    forecast_horizon = pred_config["forecast_horizon"]
    forecast_datetimes = pd.date_range(
        start=last_known_datetime + timedelta(hours=1),
        periods=forecast_horizon,
        freq='H'
    )
    results_df = pd.DataFrame({
        'Datetime': forecast_datetimes,
        f'Predicted_{pred_config["target_column"]}': unscaled_prediction.flatten()
    })
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make new forecasts with a fused model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--model", required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--data_path", required=True, help="Path to the original full data file (for fitting scalers).")
    parser.add_argument("--quantum_feature_path", default=None, help="Optional path to the pre-computed quantum features file.")
    parser.add_argument("--input_csv", required=True, help="Path to a CSV/XLSX with recent data for forecasting.")
    
    args = parser.parse_args()
    
    predictions = predict_fused(args.config, args.model, args.data_path, args.quantum_feature_path, args.input_csv)
    print("\nForecasted Predictions:")
    print(predictions)