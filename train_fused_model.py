import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
from datetime import datetime
import wandb
import argparse
from tqdm import tqdm

from utils.fused_preprocessing import load_and_prepare_fused_data
from utils.data_loader import FusedTimeSeriesDataset
from models import get_model
# We can reuse the same save/load checkpoint functions from the main train script
from train import save_checkpoint, load_checkpoint

def train_fused(config_path, data_path, quantum_feature_path, checkpoint_path, saved_model_path, resume_checkpoint_path=None):
    """
    A robust training script for the QuantumResidualTransformer, including
    W&B logging, validation, LR scheduling, and automatic tagging.
    """
    # 1. Load and Prepare Config
    with open(config_path, 'r') as f:
        file_config = yaml.safe_load(f)
    
    file_config.update({
        'data_path': data_path,
        'quantum_feature_path': quantum_feature_path,
        'checkpoint_path': checkpoint_path,
        'saved_model_path': saved_model_path
    })
    
    active_config = file_config.copy()
    
    # 2. Initialize W&B with Dynamic Tags
    if active_config.get("use_wandb", False):
        # --- NEW: Add tags for easy filtering in W&B ---
        run_tags = []
        if active_config.get('use_quantum_features', False):
            run_tags.append("quantum-enhanced")
        else:
            run_tags.append("classical-benchmark")
            
        run_name = f"{active_config.get('model_type', 'model')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            wandb.init(
                project=active_config.get("wandb_project", "default_project"), 
                entity=active_config.get("wandb_entity"), 
                config=active_config, 
                name=run_name, 
                tags=run_tags, # <-- Add the tags here
                reinit=True
            )
            active_config = wandb.config
        except Exception as e:
            print(f"Failed to initialize W&B: {e}. Running without logging.")
            active_config["use_wandb"] = False
    else:
        print("Running without W&B logging.")

    # 3. Load Data using the fused preprocessor
    data = load_and_prepare_fused_data(dict(active_config))
    
    # Create Datasets and DataLoaders
    train_dataset = FusedTimeSeriesDataset(data['X_classical_train'], data['y_train'], data.get('X_quantum_train'))
    val_dataset = FusedTimeSeriesDataset(data['X_classical_val'], data['y_val'], data.get('X_quantum_val'))
    
    train_loader = DataLoader(train_dataset, batch_size=active_config.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=active_config.get("batch_size", 32), shuffle=False)

    # 4. Initialize Model, Optimizer, Scheduler, and Criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(active_config["model_type"], dict(active_config)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=active_config.get("learning_rate", 0.0001))
    scheduler_config = active_config.get('scheduler_config', {'use_scheduler': False})
    if scheduler_config.get('use_scheduler'):
        scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_config.get('mode', 'min'), factor=scheduler_config.get('factor', 0.5), patience=scheduler_config.get('patience', 3))
    else:
        scheduler = type('DummyScheduler', (), {'step': lambda self, *args: None, 'state_dict': lambda self: None, 'load_state_dict': lambda self, *args: None})()
    criterion = nn.MSELoss()
    
    start_epoch, best_val_loss, epochs_no_improve = 0, float('inf'), 0
    print(f"Starting training for {active_config['model_type']} on {device}...")

    # 5. Main Training Loop
    for epoch in range(start_epoch, active_config.get("num_epochs", 50)):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        for batch in train_pbar:
            optimizer.zero_grad()
            
            if active_config.get('use_quantum_features', False):
                x_classical, x_quantum, y_batch = batch
                outputs = model(x_classical.to(device), x_quantum.to(device))
            else:
                x_classical, _, y_batch = batch
                outputs = model(x_classical.to(device))
            
            loss = criterion(outputs, y_batch.to(device))
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False)
            for batch in val_pbar:
                if active_config.get('use_quantum_features', False):
                    x_classical, x_quantum, y_batch = batch
                    outputs = model(x_classical.to(device), x_quantum.to(device))
                else:
                    x_classical, _, y_batch = batch
                    outputs = model(x_classical.to(device))
                loss = criterion(outputs, y_batch.to(device))
                val_loss += loss.item() * x_classical.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{active_config.get('num_epochs', 50)} - Val Loss: {epoch_val_loss:.6f} - LR: {current_lr:.6f}")
        
        scheduler.step(epoch_val_loss)
        
        if wandb.run:
            wandb.log({"epoch": epoch + 1, "val_loss": epoch_val_loss, "learning_rate": current_lr})

        # Checkpointing Logic
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            save_checkpoint(epoch, model, optimizer, scheduler, active_config, best_val_loss, epochs_no_improve, is_best=True, current_run_id=wandb.run.id if wandb.run else None)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= active_config.get("patience", 10):
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
            
    if wandb.run:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Fused Quantum Residual model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--quantum_feature_path", default=None)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--saved_model_path", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    
    train_fused(
        args.config, 
        args.data_path, 
        args.quantum_feature_path, 
        args.checkpoint_path, 
        args.saved_model_path, 
        args.resume
    )