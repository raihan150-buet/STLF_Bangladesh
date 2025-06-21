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

from utils.preprocessing import load_and_preprocess_data
from utils.data_loader import TimeSeriesDataset
from models import get_model

def save_checkpoint(epoch, model, optimizer, scheduler, config, best_val_loss, epochs_no_improve, is_best=False, current_run_id=None):
    """Saves model checkpoint, including scheduler state."""
    checkpoint_dir = config.get("checkpoint_path", "checkpoints")
    save_dir = config.get("saved_model_path", "saved_models") if is_best else checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)

    save_config = dict(config) if hasattr(config, 'as_dict') else config.copy()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': save_config,
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve,
        'wandb_run_id': current_run_id
    }
    model_identifier = f"{config.get('model_type', 'model')}_{config.get('forecasting_type', 'type')}"
    filename = f"best_{model_identifier}.pth" if is_best else f"checkpoint_{model_identifier}_epoch_{epoch}.pth"
    path_to_save = os.path.join(save_dir, filename)
    torch.save(checkpoint, path_to_save)
    print(f"Saved checkpoint: {filename} to {save_dir}")

def load_checkpoint(checkpoint_path, device):
    """Loads a training checkpoint, including scheduler state."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    loaded_config = checkpoint['config']
    model = get_model(loaded_config["model_type"], loaded_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=loaded_config.get("learning_rate", 0.001))
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_config = loaded_config.get('scheduler_config', {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config.get('mode', 'min'),
        factor=scheduler_config.get('factor', 0.1),
        patience=scheduler_config.get('patience', 10)
    )
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'epoch': checkpoint.get('epoch', 0),
        'config': loaded_config,
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'epochs_no_improve': checkpoint.get('epochs_no_improve', 0),
        'wandb_run_id': checkpoint.get('wandb_run_id')
    }

def train(config_source, data_path, checkpoint_path, saved_model_path, resume_checkpoint_path=None):
    """Main training loop with integrated LR scheduler."""
    if isinstance(config_source, dict):
        file_config = config_source
    else:
        with open(config_source, 'r') as f:
            file_config = yaml.safe_load(f)

    file_config.update({'data_path': data_path, 'checkpoint_path': checkpoint_path, 'saved_model_path': saved_model_path})
    active_config = file_config.copy()
    current_wandb_run_id = None
    if active_config.get("use_wandb", False):
        run_name = f"{active_config.get('model_type', 'model')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            wandb.init(project=active_config.get("wandb_project", "default_project"), entity=active_config.get("wandb_entity"), config=active_config, name=run_name, reinit=True)
            active_config = wandb.config
            current_wandb_run_id = wandb.run.id
            print(f"W&B Run active: {wandb.run.name} (ID: {current_wandb_run_id})")
        except Exception as e:
            print(f"Failed to initialize W&B: {e}. Running without logging.")
            active_config["use_wandb"] = False
    else:
        print("Running without W&B logging.")

    data = load_and_preprocess_data(dict(active_config))
    active_config['input_size'] = data['X_train'].shape[2]
    
    train_loader = DataLoader(TimeSeriesDataset(data["X_train"], data["y_train"]), batch_size=active_config.get("batch_size", 64), shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(data["X_val"], data["y_val"]), batch_size=active_config.get("batch_size", 64), shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    if resume_checkpoint_path:
        checkpoint_data = load_checkpoint(resume_checkpoint_path, device)
        model, optimizer, scheduler = checkpoint_data['model'].to(device), checkpoint_data['optimizer'], checkpoint_data['scheduler']
        start_epoch, best_val_loss, epochs_no_improve = checkpoint_data['epoch'] + 1, checkpoint_data['best_val_loss'], checkpoint_data['epochs_no_improve']
    else:
        model = get_model(active_config["model_type"], dict(active_config)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=active_config.get("learning_rate", 0.001))
        scheduler_config = active_config.get('scheduler_config', {'use_scheduler': False})
        if scheduler_config.get('use_scheduler'):
            # CORRECTED: Removed the 'verbose' argument
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('mode', 'min'),
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 3)
            )
            print("Using ReduceLROnPlateau learning rate scheduler.")
        else:
            scheduler = type('DummyScheduler', (), {'step': lambda self, *args: None, 'state_dict': lambda self: None, 'load_state_dict': lambda self, *args: None})()
        start_epoch, best_val_loss, epochs_no_improve = 0, float('inf'), 0
    
    print(f"Starting training for {active_config['model_type']} on {device}...")

    for epoch in range(start_epoch, active_config.get("num_epochs", 50)):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch.to(device))
                loss = criterion(outputs, y_batch.to(device))
                val_loss += loss.item() * X_batch.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{active_config.get('num_epochs', 50)} - Val Loss: {epoch_val_loss:.6f} - LR: {current_lr:.6f}")
        
        scheduler.step(epoch_val_loss)
        
        if wandb.run:
            wandb.log({"epoch": epoch + 1, "val_loss": epoch_val_loss, "learning_rate": current_lr})

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            save_checkpoint(epoch, model, optimizer, scheduler, active_config, best_val_loss, epochs_no_improve, is_best=True, current_run_id=current_wandb_run_id)
        else:
            epochs_no_improve += 1
        
        if (epoch + 1) % active_config.get("checkpoint_freq", 5) == 0 and not is_best:
             save_checkpoint(epoch, model, optimizer, scheduler, active_config, best_val_loss, epochs_no_improve, is_best=False, current_run_id=current_wandb_run_id)

        if epochs_no_improve >= active_config.get("patience", 10):
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
            
    if wandb.run:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.finish()
    
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a forecasting model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Directory for checkpoints.")
    parser.add_argument("--saved_model_path", type=str, required=True, help="Directory for best models.")
    parser.add_argument("--resume", type=str, help="Path to checkpoint .pth file to resume training.")
    args = parser.parse_args()
    train(args.config, args.data_path, args.checkpoint_path, args.saved_model_path, args.resume)
