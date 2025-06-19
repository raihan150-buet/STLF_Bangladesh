import torch
import numpy as np
import pandas as pd
import yaml
import os
from datetime import timedelta
import argparse
from utils.preprocessing import load_and_preprocess_data, CyclicalFeatureTransformer
from models import get_model

def predict(config_path, model_checkpoint_path, data_path, input_csv_path=None):
    """Makes a new forecast using a trained model and the new preprocessing logic."""
    # Load config and inject main data path
    with open(config_path, 'r') as f:
        pred_config = yaml.safe_load(f)
    pred_config['data_path'] = data_path

    # --- UPDATED LOGIC ---
    # We need the scalers and feature engineering logic from training.
    # We call load_and_preprocess_data to get the fitted scalers.
    data_info = load_and_preprocess_data(pred_config)
    feature_scaler = data_info["feature_scaler"]
    target_scaler = data_info["target_scaler"]
    
    sequence_length = pred_config["sequence_length"]
    last_known_datetime = None
    input_sequence_scaled = None

    # Determine the input sequence for the forecast
    if input_csv_path:
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input CSV for prediction not found at {input_csv_path}")
        
        # Load the recent data provided by the user
        input_df = pd.read_csv(input_csv_path)
        input_df['Datetime'] = pd.to_datetime(input_df['Datetime'])
        input_df = input_df.set_index('Datetime').sort_index()

        if len(input_df) < sequence_length:
            raise ValueError(f"Input data must have at least {sequence_length} rows.")
        
        # --- Apply the SAME feature engineering as in training ---
        df_for_pred = input_df.iloc[-sequence_length:].copy()
        last_known_datetime = df_for_pred.index[-1]

        # a. Create calendar features
        df_for_pred['hour'] = df_for_pred.index.hour
        df_for_pred['day_of_week'] = df_for_pred.index.dayofweek
        df_for_pred['day_of_month'] = df_for_pred.index.day
        df_for_pred['month'] = df_for_pred.index.month
        df_for_pred['is_weekend'] = df_for_pred['day_of_week'].isin([4, 5]).astype(int)

        # b. Lag/Rolling features - for prediction, these need to be based on the provided data
        # This requires a more complex setup where historical data is also passed.
        # For simplicity, we will assume these columns already exist in the input_csv or we will fill them.
        # A robust system would calculate these based on a larger historical context.
        print("Warning: Lag/Rolling features for prediction are assumed to be pre-calculated or will be filled with 0.")
        for col in ['demand_lag_24hr', 'demand_lag_1week', 'demand_rolling_mean_3hr', 'demand_rolling_std_24hr']:
            if col not in df_for_pred:
                df_for_pred[col] = 0

        # c. Cyclical Encoding
        cyclical_encoder = CyclicalFeatureTransformer()
        df_for_pred_encoded = cyclical_encoder.transform(df_for_pred)
        
        # d. Select and scale features
        feature_names = data_info["feature_names"]
        final_input_features = df_for_pred_encoded[feature_names].values
        input_sequence_scaled = feature_scaler.transform(final_input_features)

    else:
        # Default: Use the last sequence from the test set (already scaled)
        if len(data_info["X_test"]) > 0:
            input_sequence_scaled = data_info["X_test"][-1]
            print("Using last sequence from test data for prediction.")
            last_known_datetime = pd.Timestamp.now() # Fallback datetime
        else:
            raise ValueError("No input_csv provided and no test data available.")

    # Prepare tensor for the model
    input_tensor = torch.FloatTensor(input_sequence_scaled).unsqueeze(0) # Add batch dimension

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

    # --- UPDATED INVERSE TRANSFORM ---
    # Use the dedicated target_scaler for a clean inverse transform
    unscaled_prediction = target_scaler.inverse_transform(scaled_prediction)

    # Create results dataframe
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
    parser = argparse.ArgumentParser(description="Make new forecasts.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the original full data file for fitting the scalers.")
    parser.add_argument("--input_csv", type=str, help="Optional path to a CSV with recent data for forecasting.")
    
    args = parser.parse_args()
    
    predictions = predict(args.config, args.model, args.data_path, args.input_csv)
    print("\nForecasted Predictions:")
    print(predictions)

