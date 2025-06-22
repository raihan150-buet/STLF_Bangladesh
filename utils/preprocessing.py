import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

# This helper class remains the same
class CyclicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Encodes cyclical features using sine/cosine transformation."""
    def __init__(self):
        self.max_vals = {'hour': 23, 'day_of_week': 6, 'day_of_month': 31, 'month': 12}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for feature, max_val in self.max_vals.items():
            if feature in X_transformed.columns:
                X_transformed[f'{feature}_sin'] = np.sin(2 * np.pi * X_transformed[feature] / max_val)
                X_transformed[f'{feature}_cos'] = np.cos(2 * np.pi * X_transformed[feature] / max_val)
                X_transformed = X_transformed.drop(columns=[feature])
        return X_transformed

def load_and_preprocess_data(config):
    """
    Highly flexible data loading and preprocessing pipeline driven by a detailed config.
    """
    # 1. Load Data
    try:
        file_path = config["data_path"]
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Please use .xlsx or .csv")
    except Exception as e:
        raise FileNotFoundError(f"Could not read data file at {config['data_path']}. Error: {e}")

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # 2. Get Feature Engineering "Instructions" from Config
    fe_config = config.get('feature_engineering', {})
    use_time_features = fe_config.get('use_time_features', False)
    use_cyclical_encoding = fe_config.get('use_cyclical_encoding', False)
    use_lag_features = fe_config.get('use_lag_features', False)
    use_rolling_window_features = fe_config.get('use_rolling_window_features', False)

    # 3. Conditionally Engineer Features
    if use_time_features:
        print("Engineering time-based features...")
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = df['day_of_week'].isin([4, 5]).astype(int)

    if use_lag_features:
        print("Engineering lag features...")
        df['demand_lag_24hr'] = df[config["target_column"]].shift(24)
        df['demand_lag_1week'] = df[config["target_column"]].shift(24 * 7)

    if use_rolling_window_features:
        print("Engineering rolling window features...")
        df['demand_rolling_mean_3hr'] = df[config["target_column"]].rolling(window=3).mean()
        df['demand_rolling_std_24hr'] = df[config["target_column"]].rolling(window=24).std()

    # Handle NaN values created by feature engineering
    df = df.fillna(method='bfill').fillna(method='ffill')

    # 4. Select Final Features for the Model
    target_col = config["target_column"]
    features_to_use = [target_col] # Start with the target column
    
    # Add engineered features if their flags were set
    if use_time_features:
        features_to_use.extend(['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend'])
    if use_lag_features:
        features_to_use.extend(['demand_lag_24hr', 'demand_lag_1week'])
    if use_rolling_window_features:
        features_to_use.extend(['demand_rolling_mean_3hr', 'demand_rolling_std_24hr'])

    # Add external features if in multivariate mode
    if config["forecasting_type"] == "multivariate":
        external_features = config.get("features", [])
        features_to_use.extend(external_features)
        
    # Remove duplicates while preserving order
    features_to_use = sorted(set(features_to_use), key=features_to_use.index)
    data = df[features_to_use].copy()
    
    # 5. Conditionally Apply Cyclical Encoding
    if use_cyclical_encoding:
        print("Applying cyclical encoding...")
        cyclical_encoder = CyclicalFeatureTransformer()
        data = cyclical_encoder.transform(data)
        
    # --- The rest of the pipeline remains the same ---
    
    # 6. Create Sequences
    feature_data = data.drop(columns=[target_col]).values
    target_data = data[[target_col]].values
    X, y, = [], []
    for i in range(len(data) - config["sequence_length"] - config["forecast_horizon"] + 1):
        X.append(feature_data[i : i + config["sequence_length"]])
        y.append(target_data[i + config["sequence_length"] : i + config["sequence_length"] + config["forecast_horizon"]])
    X, y = np.array(X), np.array(y).reshape(-1, config["forecast_horizon"])
    
    # 7. Temporal Split
    test_size = int(len(X) * config["test_size"])
    val_size = int(len(X) * config["val_size"])
    X_train, y_train = X[:-(test_size+val_size)], y[:-(test_size+val_size)]
    X_val, y_val = X[-(test_size+val_size):-test_size], y[-(test_size+val_size):-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    # 8. Separate Scaling
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = feature_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = feature_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train = target_scaler.fit_transform(y_train)
    y_val = target_scaler.transform(y_val)
    y_test = target_scaler.transform(y_test)

    # Update input_size in config based on the final number of features
    config["input_size"] = X_train.shape[2]
    
    print(f"Data preprocessed. Final input feature count: {config['input_size']}")
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_names": data.drop(columns=[target_col]).columns.tolist()
    }
