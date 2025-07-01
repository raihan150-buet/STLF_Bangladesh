import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import pandas as pd
import yaml
import argparse
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def create_quantum_features(config_path, data_path, output_path):
    """
    Loads a dataset, applies classical feature engineering, and then transforms a
    user-defined subset of features using a quantum circuit, saving the result.
    """
    print("--- Starting Row-by-Row Quantum Feature Extraction ---")
    
    # 1. Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Select the classical features to be transformed from the config
    features_for_quantum_input = config.get("features_for_quantum")
    if not features_for_quantum_input or not isinstance(features_for_quantum_input, list):
        raise ValueError("'features_for_quantum' list not found or is empty in config file.")
    
    # --- DYNAMIC QUANTUM CONFIGURATION ---
    n_qubits = len(features_for_quantum_input)
    q_depth = config.get('q_depth', 2)
    print(f"Dynamically configuring a {n_qubits}-qubit circuit based on config.")
    
    # 3. Define the Quantum Circuit
    dev = qml.device("lightning.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def quantum_feature_extractor_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (q_depth, n_qubits)}
    qlayer = qml.qnn.TorchLayer(quantum_feature_extractor_circuit, weight_shapes)

    # 4. Load data and create necessary classical features
    print(f"Loading data and selecting features: {features_for_quantum_input}")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path, engine='openpyxl')
    else:
        df = pd.read_csv(data_path)
    
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()

    # Prepare the selected classical data
    classical_input_data = df[features_for_quantum_input].copy()
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    classical_input_scaled = scaler.fit_transform(classical_input_data)
    
    # 5. Process all time steps through the dynamically sized quantum circuit
    print(f"Extracting {n_qubits} quantum features for each time step...")
    all_quantum_features = []
    
    for row in tqdm(torch.tensor(classical_input_scaled, dtype=torch.float32)):
        with torch.no_grad():
            q_features = qlayer(row).numpy()
        all_quantum_features.append(q_features)

    # 6. Save the new features
    output_columns = [f'q_feat_of_{col}' for col in features_for_quantum_input]
    quantum_features_df = pd.DataFrame(all_quantum_features, 
                                       columns=output_columns,
                                       index=df.index)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    quantum_features_df.to_csv(output_path)
    print(f"\nSuccessfully saved {len(all_quantum_features)} quantum feature vectors to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible Quantum Feature Extractor")
    parser.add_argument("--config", required=True, help="Path to a YAML config file with quantum settings.")
    parser.add_argument("--data_path", required=True, help="Path to the original input data file.")
    parser.add_argument("--output_path", required=True, help="Path to save the resulting quantum features CSV.")
    
    args = parser.parse_args()
    create_quantum_features(args.config, args.data_path, args.output_path)