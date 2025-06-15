import wandb
import yaml
import os
import argparse
from functools import partial
from train import train

def run_sweep_agent(base_config):
    """Wrapper function called by the W&B agent for each trial."""
    # W&B automatically injects the sweep's hyperparameter choices.
    # We call our main train function, passing the base config.
    # The train function will initialize W&B, which merges the sweep config.
    print("Sweep Agent: Starting a new training run.")
    # The `train` function now expects paths. Since this function is called by the
    # agent without arguments, we must get the paths from the pre-configured base_config.
    train(
        config_source=base_config,
        data_path=base_config['data_path'],
        checkpoint_path=base_config['checkpoint_path'],
        saved_model_path=base_config['saved_model_path']
    )
    print("Sweep Agent: Run finished.")

def main_sweep_controller(config_path, data_path, checkpoint_path, saved_model_path, count):
    """Initializes a W&B sweep and starts an agent."""
    # Load base config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Inject runtime paths into the config. This is crucial.
    base_config['data_path'] = data_path
    base_config['checkpoint_path'] = checkpoint_path
    base_config['saved_model_path'] = saved_model_path

    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'min': 0.00005, 'max': 0.005, 'distribution': 'log_uniform_values'},
            'batch_size': {'values': [32, 64, 128]},
            'hidden_size': {'values': [32, 64, 128, 256]},
            'num_layers': {'values': [1, 2, 3]},
            'dropout': {'min': 0.05, 'max': 0.4, 'distribution': 'uniform'},
            'sequence_length': {'values': [24, 48, 96, 168]},
        }
    }
    
    wandb_project = base_config.get('wandb_project', 'default_sweeps')
    wandb_entity = base_config.get('wandb_entity')

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=wandb_project, entity=wandb_entity)
    print(f"Sweep initialized. ID: {sweep_id}")

    # Create a partial function to pass the configured base_config to the agent's target function
    agent_function = partial(run_sweep_agent, base_config=base_config)
    
    # Start the agent directly in this script for convenience in Colab
    print(f"Starting W&B agent to run {count} trials...")
    wandb.agent(sweep_id, function=agent_function, count=count)
    print("Sweep agent has completed all trials.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize and run a W&B hyperparameter sweep.")
    parser.add_argument("--config", type=str, required=True, help="Path to base YAML config.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Directory for checkpoints.")
    parser.add_argument("--saved_model_path", type=str, required=True, help="Directory for best models.")
    parser.add_argument("--count", type=int, default=5, help="Number of sweep trials to run.")

    args = parser.parse_args()
    
    main_sweep_controller(
        args.config,
        args.data_path,
        args.checkpoint_path,
        args.saved_model_path,
        args.count
    )
