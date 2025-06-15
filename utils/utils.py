import yaml
import torch
import os

# Note: The main train.py script has its own more detailed checkpointing functions.
# These are simpler, legacy versions. They are kept here for completeness but
# are not actively called by the latest version of train.py.

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint_simple(model, optimizer, epoch, path):
    """A simpler checkpoint saving function."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint_simple(model, optimizer, path):
    """A simpler checkpoint loading function."""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0
