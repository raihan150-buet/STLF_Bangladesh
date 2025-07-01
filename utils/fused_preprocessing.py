import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_fused_data(config):
    """
    Loads, fuses, sequences, and correctly scales classical and quantum features.
    """
    # 1. Load Original and Quantum Data
    try:
        if config["data_path"].endswith('.xlsx'):
            df_classical = pd.read_excel(config["data_path"], engine='openpyxl')
        else:
            df_classical = pd.read_csv(config["data_path"])
        df_classical['Datetime'] = pd.to_datetime(df_classical['Datetime']).dt.round('min')
        df_classical = df_classical.set_index('Datetime').sort_index()

        if config.get('use_quantum_features', False):
            print("Loading pre-computed quantum features...")
            df_quantum = pd.read_csv(config["quantum_feature_path"])
            df_quantum['Datetime'] = pd.to_datetime(df_quantum['Datetime']).dt.round('min')
            df_quantum = df_quantum.set_index('Datetime').sort_index()
            
            if len(df_classical) != len(df_quantum):
                raise ValueError(f"Classical ({len(df_classical)}) and quantum ({len(df_quantum)}) feature files have different lengths.")
            
            df_full = pd.concat([df_classical, df_quantum], axis=1)
        else:
            print("Running in classical-only mode. Skipping quantum features.")
            df_full = df_classical
    except Exception as e:
        raise FileNotFoundError(f"Error loading or merging data files: {e}")

    df_full['is_weekend'] = df_full.index.dayofweek.isin([4, 5]).astype(int)

    # 2. Select Features and Create Sequences
    target_col = config["target_column"]
    classical_features_to_use = config.get("classical_features", [])
    
    X_classical_data = df_full[classical_features_to_use].values
    y_data = df_full[[target_col]].values
    
    if config.get('use_quantum_features', False):
        quantum_features_to_use = [col for col in df_full.columns if 'q_feat' in col]
        X_quantum_data = df_full[quantum_features_to_use].values

    X_classical, X_quantum, y = [], [], []
    seq_len = config["sequence_length"]
    horizon = config["forecast_horizon"]
    for i in range(len(df_full) - seq_len - horizon + 1):
        X_classical.append(X_classical_data[i : i + seq_len])
        if config.get('use_quantum_features', False):
            X_quantum.append(X_quantum_data[i : i + seq_len])
        y.append(y_data[i + seq_len : i + seq_len + horizon])
    
    X_classical, y = np.array(X_classical), np.array(y).reshape(-1, horizon)
    if config.get('use_quantum_features', False):
        X_quantum = np.array(X_quantum)
    
    # 3. Temporal Split
    test_size = int(len(y) * config["test_size"])
    val_size = int(len(y) * config["val_size"])
    X_classical_train, X_classical_val, X_classical_test = X_classical[:-(test_size+val_size)], X_classical[-(test_size+val_size):-test_size], X_classical[-test_size:]
    y_train, y_val, y_test = y[:-(test_size+val_size)], y[-(test_size+val_size):-test_size], y[-test_size:]
    
    # --- 4. ADDED SCALING LOGIC ---
    # Initialize scalers
    classical_feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Reshape classical features to 2D for scaler, fit ONLY on training data
    n_samples_train, seq_len_train, n_features_train = X_classical_train.shape
    X_classical_train_reshaped = X_classical_train.reshape(-1, n_features_train)
    classical_feature_scaler.fit(X_classical_train_reshaped)

    # Transform all classical feature sets
    X_classical_train = classical_feature_scaler.transform(X_classical_train_reshaped).reshape(X_classical_train.shape)
    X_classical_val = classical_feature_scaler.transform(X_classical_val.reshape(-1, n_features_train)).reshape(X_classical_val.shape)
    X_classical_test = classical_feature_scaler.transform(X_classical_test.reshape(-1, n_features_train)).reshape(X_classical_test.shape)

    # Fit and transform target variable separately
    target_scaler.fit(y_train)
    y_train = target_scaler.transform(y_train)
    y_val = target_scaler.transform(y_val)
    y_test = target_scaler.transform(y_test)
    
    result = {
        "X_classical_train": X_classical_train, "y_train": y_train,
        "X_classical_val": X_classical_val, "y_val": y_val,
        "X_classical_test": X_classical_test, "y_test": y_test,
        "classical_feature_scaler": classical_feature_scaler,
        "target_scaler": target_scaler,
    }
    
    if config.get('use_quantum_features', False):
        # Also scale the quantum features if they exist
        quantum_feature_scaler = MinMaxScaler()
        X_quantum_train, X_quantum_val, X_quantum_test = X_quantum[:-(test_size+val_size)], X_quantum[-(test_size+val_size):-test_size], X_quantum[-test_size:]
        
        n_samples_q, seq_len_q, n_features_q = X_quantum_train.shape
        X_quantum_train_reshaped = X_quantum_train.reshape(-1, n_features_q)
        quantum_feature_scaler.fit(X_quantum_train_reshaped)
        
        X_quantum_train = quantum_feature_scaler.transform(X_quantum_train_reshaped).reshape(X_quantum_train.shape)
        X_quantum_val = quantum_feature_scaler.transform(X_quantum_val.reshape(-1, n_features_q)).reshape(X_quantum_val.shape)
        X_quantum_test = quantum_feature_scaler.transform(X_quantum_test.reshape(-1, n_features_q)).reshape(X_quantum_test.shape)
        
        result.update({
            "X_quantum_train": X_quantum_train,
            "X_quantum_val": X_quantum_val,
            "X_quantum_test": X_quantum_test,
            "quantum_feature_scaler": quantum_feature_scaler,
        })
    
    return result