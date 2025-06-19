import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Helper class for cyclical feature engineering
class CyclicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Encodes cyclical features (e.g., hour, day, month) using sine/cosine transformation."""
    def __init__(self):
        # Using standard max values for time features
        self.max_vals = {
            'hour': 23, 
            'day_of_week': 6, 
            'day_of_month': 31, 
            'month': 12
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for feature, max_val in self.max_vals.items():
            if feature in X_transformed.columns:
                # Create sin/cos features and drop the original
                X_transformed[f'{feature}_sin'] = np.sin(2 * np.pi * X_transformed[feature] / max_val)
                X_transformed[f'{feature}_cos'] = np.cos(2 * np.pi * X_transformed[feature] / max_val)
                X_transformed = X_transformed.drop(columns=[feature])
        return X_transformed

def load_and_preprocess_data(config):
    """
    Enhanced data loading and preprocessing with advanced feature engineering.
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

    # 2. Initial Datetime and Index Setup
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()

    # 3. Advanced Feature Engineering
    target_col = config["target_column"]
    
    # a. Calendar and Holiday Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([4, 5]).astype(int) # Friday=4, Saturday=5
    # The original 'Holiday' column is kept as a separate feature

    # b. Lag Features
    df['demand_lag_24hr'] = df[target_col].shift(24)
    df['demand_lag_1week'] = df[target_col].shift(24 * 7)

    # c. Rolling Window Features
    df['demand_rolling_mean_3hr'] = df[target_col].rolling(window=3).mean()
    df['demand_rolling_std_24hr'] = df[target_col].rolling(window=24).std()

    # d. Handle NaN values created by lags and rolling windows
    # We use backfill to avoid losing data at the start, then forward fill for any remaining
    df = df.fillna(method='bfill').fillna(method='ffill')

    # 4. Select Features based on Config
    base_features = [target_col]
    
    # These engineered features will be added to both univariate and multivariate models
    engineered_features = ['is_weekend', 'demand_lag_24hr', 'demand_lag_1week', 
                             'demand_rolling_mean_3hr', 'demand_rolling_std_24hr']
    
    # These time features will be cyclically encoded
    time_features_to_encode = ['hour', 'day_of_week', 'day_of_month', 'month']
    
    if config["forecasting_type"] == "univariate":
        features_to_use = base_features + engineered_features + time_features_to_encode
    else:
        # For multivariate, use everything plus the external features from the config
        external_features = config.get("features", []) # e.g., ['Heat_Index_C', 'Generation(MW)']
        features_to_use = base_features + engineered_features + time_features_to_encode + external_features

    data = df[features_to_use].copy()
    
    # 5. Apply Cyclical Encoding
    cyclical_encoder = CyclicalFeatureTransformer()
    data = cyclical_encoder.transform(data)
    
    # 6. Create Sequences
    # Separate target from features before creating sequences
    feature_data = data.drop(columns=[target_col]).values
    target_data = data[[target_col]].values

    X, y = [], []
    seq_len = config["sequence_length"]
    horizon = config["forecast_horizon"]

    for i in range(len(data) - seq_len - horizon + 1):
        X.append(feature_data[i : i + seq_len])
        y.append(target_data[i + seq_len : i + seq_len + horizon])
    
    X, y = np.array(X), np.array(y).reshape(-1, horizon)
    
    # 7. Temporal Train-Validation-Test Split
    test_size = int(len(X) * config["test_size"])
    val_size = int(len(X) * config["val_size"])
    
    X_train_val, y_train_val = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]
    
    X_train, y_train = X_train_val[:-val_size], y_train_val[:-val_size]
    X_val, y_val = X_train_val[-val_size:], y_train_val[-val_size:]

    # 8. Separate Scaling for Features and Target
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Reshape features to 2D for scaler, fit ONLY on training data
    n_samples_train, seq_len_train, n_features_train = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features_train)
    feature_scaler.fit(X_train_reshaped)

    # Transform all feature sets
    X_train = feature_scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val = feature_scaler.transform(X_val.reshape(-1, n_features_train)).reshape(X_val.shape)
    X_test = feature_scaler.transform(X_test.reshape(-1, n_features_train)).reshape(X_test.shape)

    # Fit and transform target variable separately
    target_scaler.fit(y_train)
    y_train = target_scaler.transform(y_train)
    y_val = target_scaler.transform(y_val)
    y_test = target_scaler.transform(y_test)

    # Update input_size in config based on the final number of features after encoding
    config["input_size"] = X_train.shape[2]
    
    print(f"Data preprocessed successfully. Final input feature count: {config['input_size']}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_names": data.drop(columns=[target_col]).columns.tolist()
    }
