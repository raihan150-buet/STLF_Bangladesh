import torch
import numpy as np
import pandas as pd
import yaml
import os
from datetime import timedelta
import argparse
from utils.preprocessing import load_and_preprocess_data
from models import get_model

def predict(config_path, model_checkpoint_path, data_path, input_csv_path=None):
    """Makes a new forecast using a trained model."""
    # Load config and inject main data path
    with open(config_path, 'r') as f:
        pred_config = yaml.safe_load(f)
    pred_config['data_path'] = data_path

    # Load the scaler and feature names by processing the original data
    # This ensures the scaler is identical to the one used in training.
    data_info = load_and_preprocess_data(pred_config)
    scaler = data_info["scaler"]
    scaler_feature_names = data_info["feature_names"]
    
    sequence_length = pred_config["sequence_length"]
    last_known_datetime = None
    input_sequence = None

    # Determine the input sequence for the forecast
    if input_csv_path:
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input CSV for prediction not found at {input_csv_path}")
        input_df = pd.read_csv(input_csv_path, index_col='Datetime', parse_dates=True).sort_index()
        if len(input_df) < sequence_length:
            raise ValueError(f"Input data must have at least {sequence_length} rows.")
        
        relevant_input_df = input_df.iloc[-sequence_length:]
        last_known_datetime = relevant_input_df.index[-1]
        
        # Ensure all required columns are present for the scaler
        data_for_scaling = pd.DataFrame(columns=scaler_feature_names, index=relevant_input_df.index)
        data_for_scaling.update(relevant_input_df)
        data_for_scaling.fillna(0, inplace=True) # Or use a more sophisticated fill method

        input_sequence = scaler.transform(data_for_scaling[scaler_feature_names])
    else:
        # Default: Use the last sequence from the test set
        if len(data_info["X_test"]) > 0:
            input_sequence = data_info["X_test"][-1]
            print("Using last sequence from test data for prediction.")
            # This is a fallback; a real application should provide fresh data
            last_known_datetime = pd.Timestamp.now()
        else:
            raise ValueError("No input_csv provided and no test data available.")

    # Prepare tensor for the model
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0) # Add batch dimension

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    model = get_model(model_config["model_type"], model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        scaled_prediction = model(input_tensor.to(device)).cpu().numpy()

    # Inverse transform the prediction
    num_scaler_features = scaler.n_features_in_
    target_idx = scaler_feature_names.index(pred_config["target_column"])
    
    pred_reshaped = scaled_prediction.reshape(-1, 1)
    dummy_array = np.zeros((pred_reshaped.shape[0], num_scaler_features))
    dummy_array[:, target_idx] = pred_reshaped[:, 0]
    unscaled_prediction = scaler.inverse_transform(dummy_array)[:, target_idx]

    # Create results dataframe
    forecast_horizon = pred_config["forecast_horizon"]
    forecast_datetimes = pd.date_range(
        start=last_known_datetime + timedelta(hours=1),
        periods=forecast_horizon,
        freq='H'
    )
    results_df = pd.DataFrame({
        'Datetime': forecast_datetimes,
        f'Predicted_{pred_config["target_column"]}': unscaled_prediction
    })
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make new forecasts.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the original data file for fitting the scaler.")
    parser.add_argument("--input_csv", type=str, help="Optional path to a CSV with recent data for forecasting.")
    
    args = parser.parse_args()
    
    predictions = predict(args.config, args.model, args.data_path, args.input_csv)
    print("\nForecasted Predictions:")
    print(predictions)
