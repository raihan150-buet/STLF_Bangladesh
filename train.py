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
from typing import Dict, Optional, Union

from utils.preprocessing import load_and_preprocess_data
from utils.data_loader import TimeSeriesDataset
from models import get_model

def get_default_device():
    """Automatically select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class TrainingLogger:
    """Handles logging to both console and W&B with consistent formatting."""
    def __init__(self, use_wandb: bool = False):
        self.use_wandb = use_wandb
        
    def log(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to both console and W&B."""
        log_str = " | ".join(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                            for k, v in metrics.items())
        print(log_str)
        
        if self.use_wandb and wandb.run:
            wandb.log(metrics, step=step)

def create_filename(config: Dict, epoch: Optional[int] = None, 
                   is_best: bool = False, run_id: Optional[str] = None) -> str:
    """Generate standardized filenames for checkpoints."""
    model_name = config.get("model_type", "model")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if is_best:
        return f"best_{model_name}_{run_id or timestamp}.pth"
    return f"checkpoint_{model_name}_epoch{epoch}_{run_id or timestamp}.pth"

class CheckpointManager:
    """Handles saving and loading model checkpoints with full state preservation."""
    def __init__(self, checkpoint_dir: str, saved_models_dir: str):
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(saved_models_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.saved_models_dir = saved_models_dir
        
    def save(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, 
             scheduler, config: Dict, metrics: Dict, is_best: bool = False, 
             run_id: Optional[str] = None) -> str:
        """Save model checkpoint with all training state."""
        save_dir = self.saved_models_dir if is_best else self.checkpoint_dir
        filename = create_filename(config, epoch, is_best, run_id)
        path = os.path.join(save_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'metrics': metrics,
            'wandb_run_id': run_id
        }, path)
        
        return path
    
    @staticmethod
    def load(path: str, device: torch.device) -> Dict:
        """Load checkpoint and map to specified device."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        checkpoint = torch.load(path, map_location=device)
        
        # Handle legacy checkpoints
        if 'best_val_loss' in checkpoint:  # Backward compatibility
            checkpoint['metrics'] = {'val_loss': checkpoint['best_val_loss']}
            
        return checkpoint

class EarlyStopper:
    """Handles early stopping based on validation metrics."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = float('inf')
        
    def should_stop(self, current_metric: float) -> bool:
        if current_metric < (self.best_metric - self.min_delta):
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def initialize_wandb(config: Dict, run_id: Optional[str] = None) -> bool:
    """Initialize W&B logging with optional run resuming."""
    if not config.get("use_wandb", False):
        return False
        
    try:
        wandb.init(
            project=config.get("wandb_project"),
            entity=config.get("wandb_entity"),
            config=config,
            id=run_id,
            resume="allow" if run_id else None,
            reinit=True
        )
        print(f"W&B Run: {wandb.run.name} (ID: {wandb.run.id})")
        return True
    except Exception as e:
        print(f"W&B initialization failed: {e}")
        return False

def train_epoch(model: nn.Module, dataloader: DataLoader, 
               optimizer: optim.Optimizer, criterion: nn.Module, 
               device: torch.device, clip_grad: Optional[float] = None) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        pbar.set_postfix(loss=f"{loss.item():.6f}")
        
    return total_loss / len(dataloader.dataset)

def validate(model: nn.Module, dataloader: DataLoader, 
            criterion: nn.Module, device: torch.device) -> float:
    """Validate model performance."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Validation", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            
    return total_loss / len(dataloader.dataset)

def train(config_file: str, data_path: str, checkpoint_dir: str, 
         saved_models_dir: str, resume_from: Optional[str] = None):
    """Main training procedure with full training lifecycle management."""
    # Load configuration
    with open(config_file) as f:
        config = yaml.safe_load(f)
        
    config['data_path'] = data_path
    config['checkpoint_dir'] = checkpoint_dir
    config['saved_models_dir'] = saved_models_dir
    # Initialize device
    device = get_default_device()
    print(f"Using device: {device}")
    
    # Handle W&B resuming
    wandb_run_id = None
    if resume_from:
        checkpoint = CheckpointManager.load(resume_from, 'cpu')
        wandb_run_id = checkpoint.get('wandb_run_id')
    
    # Initialize W&B
    use_wandb = initialize_wandb(config, wandb_run_id)
    if use_wandb:
        config = wandb.config
    
    # Load and prepare data
    data = load_and_preprocess_data(config)
    train_loader = DataLoader(
        TimeSeriesDataset(data["X_train"], data["y_train"]),
        batch_size=config.get("batch_size", 64),
        shuffle=True
    )
    val_loader = DataLoader(
        TimeSeriesDataset(data["X_val"], data["y_val"]),
        batch_size=config.get("batch_size", 64),
        shuffle=False
    )
    
    # Initialize model and training components
    if resume_from:
        checkpoint = CheckpointManager.load(resume_from, device)
        model = get_model(checkpoint['config']["model_type"], checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        scheduler = ReduceLROnPlateau(optimizer, **config.get('scheduler_config', {}))
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_metrics = checkpoint.get('metrics', {'val_loss': float('inf')})
    else:
        model = get_model(config["model_type"], config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
        scheduler = ReduceLROnPlateau(optimizer, **config.get('scheduler_config', {}))
        start_epoch = 0
        best_metrics = {'val_loss': float('inf')}
    
    criterion = nn.MSELoss()
    checkpoint_manager = CheckpointManager(checkpoint_dir, saved_models_dir)
    early_stopper = EarlyStopper(patience=config.get("patience", 10))
    logger = TrainingLogger(use_wandb)
    
    # Main training loop
    for epoch in range(start_epoch, config.get("num_epochs", 50)):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_grad=config.get("grad_clip", None)
        )
        
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        }
        logger.log(metrics)
        
        # Save checkpoints
        is_best = val_loss < best_metrics['val_loss']
        if is_best:
            best_metrics = metrics.copy()
            
        checkpoint_path = checkpoint_manager.save(
            epoch, model, optimizer, scheduler, config,
            metrics, is_best, wandb.run.id if use_wandb else None
        )
        
        # Early stopping
        if early_stopper.should_stop(val_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Finalize W&B run
    if use_wandb:
        wandb.summary.update(best_metrics)
        if os.path.exists(checkpoint_path):
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                metadata=config
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a time series forecasting model.")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--data_path", required=True, help="Path to training data")
    parser.add_argument("--checkpoint_path", required=True, help="Directory for checkpoints")
    parser.add_argument("--saved_model_path", required=True, help="Directory for best models")
    parser.add_argument("--resume", help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    train(
        args.config,
        args.data_path,
        args.checkpoint_path,
        args.saved_model_path,
        args.resume
    )