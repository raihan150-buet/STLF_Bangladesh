import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(config):
    """Loads, preprocesses, and splits the time series data."""
    # Load data from the path provided in the config
    try:
        df = pd.read_excel(config["data_path"])
    except Exception as e:
        raise FileNotFoundError(f"Could not read data file at {config['data_path']}. Error: {e}")

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # Feature selection based on forecasting type
    if config["forecasting_type"] == "univariate":
        features_to_use = [config["target_column"]]
        # Mutate the config dict to set input_size for the model
        config["input_size"] = 1
    else:
        features_to_use = [config["target_column"]] + config["features"]
        config["input_size"] = len(features_to_use)
    
    data = df[features_to_use].copy()
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    seq_len = config["sequence_length"]
    horizon = config["forecast_horizon"]
    target_col_idx = data.columns.get_loc(config["target_column"])
    
    for i in range(len(scaled_data) - seq_len - horizon + 1):
        X.append(scaled_data[i : i + seq_len])
        y.append(scaled_data[i + seq_len : i + seq_len + horizon, target_col_idx])
    
    X, y = np.array(X), np.array(y)
    
    # Split data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"], shuffle=False
    )
    
    # Adjust validation split size relative to the remaining training data
    val_split_ratio = config["val_size"] / (1 - config["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split_ratio, random_state=config["random_state"], shuffle=False
    )
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "feature_names": data.columns.tolist()
    }
