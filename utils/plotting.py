import matplotlib.pyplot as plt
import numpy as np
import os # For checking save_path directory

def plot_training_history(train_losses, val_losses, save_path=None, show_plot=False):
    """
    Plots training and validation loss history.
    Args:
        train_losses (list or np.array): List of training losses per epoch.
        val_losses (list or np.array): List of validation losses per epoch.
        save_path (str, optional): Path to save the plot image. Defaults to None.
        show_plot (bool, optional): Whether to call plt.show(). Defaults to False.
                                    Set to True for interactive display.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='royalblue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='darkorange', linewidth=2)
    plt.title('Model Training History', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=14) # Assuming MSE, adjust if different
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Training history plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving training history plot to {save_path}: {e}")
    
    if show_plot:
        plt.show()
    else:
        plt.close() # Close the figure if not showing, to free memory

def plot_predictions(y_true, y_pred, n_samples=3, forecast_horizon=None, save_path=None, show_plot=False):
    """
    Plots actual vs. predicted values for a few samples.
    This function is designed to draw on the CURRENT ACTIVE Matplotlib figure if one is provided
    by the caller (e.g., for W&B logging), or create a new one.

    Args:
        y_true (np.array): True target values. Shape (num_total_samples, [horizon_steps]).
        y_pred (np.array): Predicted values. Shape (num_total_samples, [horizon_steps]).
        n_samples (int): Number of sample series to plot from the beginning of y_true/y_pred.
        forecast_horizon (int, optional): Length of the forecast horizon for x-axis labeling.
                                         If None, assumes y_true/y_pred are single step or uses their length.
        save_path (str, optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to call plt.show(). Defaults to False.
                                    Set to True for interactive display.
    """
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: y_true or y_pred is empty. Cannot plot predictions.")
        return

    num_total_samples = y_true.shape[0]
    n_to_plot = min(n_samples, num_total_samples)

    if n_to_plot == 0:
        print("No samples available to plot predictions.")
        return

    # Determine if the figure context is managed externally or create a new one
    # If called from evaluate.py for W&B, evaluate.py creates the figure.
    # If called standalone, this function creates the figure.
    if plt.gcf().get_axes(): # Check if there's an active figure with axes
        fig = plt.gcf()
        # Clear existing axes if any, to redraw, or assume caller wants to add subplots.
        # For simplicity, let's assume if fig has axes, we add to it or it's set up.
        # This part can be tricky. Safest is often for this function to always create its own figure
        # if not explicitly passed an `ax` or `fig` object.
        # For now, let's assume it creates its own figure if show_plot is True,
        # otherwise, it draws on the current figure.
        if not show_plot: # Drawing on externally managed figure (e.g. for W&B)
             fig.set_size_inches(15, 5 * n_to_plot) # Resize if needed
        else: # Standalone plotting, create new figure
             fig, axes = plt.subplots(n_to_plot, 1, figsize=(15, 5 * n_to_plot), squeeze=False)
    else:
        fig, axes = plt.subplots(n_to_plot, 1, figsize=(15, 5 * n_to_plot), squeeze=False)


    for i in range(n_to_plot):
        ax = axes[i, 0] if not show_plot and plt.gcf().get_axes() else (axes[i,0] if n_to_plot > 1 else axes[0]) # Handle subplot array

        actual_series = y_true[i]
        predicted_series = y_pred[i]
        
        # Determine x-axis (time steps or horizon steps)
        if forecast_horizon and actual_series.ndim == 1 and len(actual_series) == forecast_horizon:
            x_axis = np.arange(1, forecast_horizon + 1)
            xlabel = 'Forecast Horizon Step'
        elif actual_series.ndim == 1:
            x_axis = np.arange(len(actual_series))
            xlabel = 'Time Step Index'
        else: # Default to index if shape is unexpected
            x_axis = np.arange(actual_series.shape[-1]) # Assumes last dim is time/horizon
            xlabel = 'Step'


        ax.plot(x_axis, actual_series, label='Actual', color='dodgerblue', marker='.', linestyle='-')
        ax.plot(x_axis, predicted_series, label='Predicted', color='orangered', marker='x', linestyle='--')
        
        ax.set_title(f'Sample {i+1} - Actual vs. Predicted', fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Value', fontsize=12) # Adjust label as needed
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle('Model Predictions vs. Actual Values', fontsize=18, y=1.03 if n_to_plot > 1 else 1.0) # Adjust y for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.98 if n_to_plot > 1 else 0.95]) # Adjust rect for suptitle

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving predictions plot to {save_path}: {e}")

    if show_plot:
        plt.show()
    elif not plt.gcf().get_axes() : # If this function created the figure and not showing, close it
        plt.close(fig)
    # If drawing on an external figure and not showing, do not close it here. Caller manages it.