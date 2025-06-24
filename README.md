# MSc Thesis: Quantum-Enhanced LSTM for Demand Forecasting

This repository contains the source code for an MSc thesis project focused on short-term electricity demand forecasting. It explores and compares various deep learning models, including classical models like LSTM, TCN, and Transformer, as well as a novel Quantum-Enhanced LSTM (QEnhancedLSTM).



## Core Features

- **Modular Codebase:** A clean separation of concerns with dedicated folders for models, utilities, and configurations.
- **Dynamic Configuration:** Experiments are controlled via `.yaml` files, allowing for easy changes without altering the source code.
- **Reproducible Workflow:** Designed to run seamlessly in Google Colab, ensuring a consistent environment for all experiments.
- **Advanced Experiment Tracking:** Integrated with Weights & Biases to log metrics, configurations, model checkpoints, and output plots automatically.
- **Hyperparameter Optimization:** Includes a script to run automated hyperparameter sweeps using W&B Sweeps.

## Project Structure

```
.
├── configs/              # Experiment configuration files (.yaml)
│   ├── base_config.yaml
│   ├── multivariate_config.yaml
│   ├── q_config.yaml
│   └── univariate_config.yaml
├── models/               # Model architecture definitions (.py)
│   ├── __init__.py
│   ├── base_model.py
│   ├── lstm.py
│   └── ...
├── utils/                # Helper scripts for data processing, metrics, etc.
│   ├── data_loader.py
│   ├── metrics.py
│   ├── plotting.py
│   ├── preprocessing.py
│   └── utils.py
├── .gitignore            # Specifies which files Git should ignore
├── evaluate.py           # Script to evaluate a trained model
├── predict.py            # Script to make new predictions
├── README.md             # This file
├── requirements.txt      # Python package dependencies
├── sweep.py              # Script to run W&B hyperparameter sweeps
└── train.py              # Main script for model training
```

## Workflow Guide

This project is designed to be run from the `MSc_Thesis_Master_Workflow.ipynb` notebook in Google Colab.

### One-Time Setup

1.  **GitHub:** Clone this repository.
2.  **Google Drive:** Create a main folder (e.g., `MSc_Thesis`) and within it, create three subfolders: `data`, `checkpoints`, and `saved_models`.
3.  **Data:** Upload your dataset (e.g., `cleaned_with_seasonality.xlsx`) to the `MSc_Thesis/data/` folder on Google Drive.

### Running Experiments

1.  **Open the Master Notebook:** Open `MSc_Thesis_Master_Workflow.ipynb` in Google Colab.
2.  **Configure Paths:** In Cell 2 of the notebook, update the `DRIVE_ROOT_DIR` and `GIT_REPO_NAME` variables to match your setup.
3.  **Run Setup Cells:** Execute cells 1 through 5 to mount your Drive, configure paths, pull the latest code, install dependencies, and log in to W&B.
4.  **Execute an Action:** Choose one of the "ACTION" cells at the bottom of the notebook to train, evaluate, or run a hyperparameter sweep.

This structured workflow ensures that your code is version-controlled, your large files are managed separately, and all your experiments are logged and reproducible.
