import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_rl_learning_curves(models_data, output_path, fig_name, xlabel="Training Steps", ylabel="Average Reward"):
    """
    Plots learning curves for multiple reinforcement learning models.

    Parameters:
    - models_data: dict, where each key is a model name and the value is a dict with:
        - 'steps': 1D array-like, x-axis values (training steps)
        - 'mean': 1D array-like, average reward at each step
        - 'std': 1D array-like, standard deviation of reward at each step
    - title: str, title of the plot
    - xlabel: str, label for the x-axis
    - ylabel: str, label for the y-axis
    """

    plt.figure(figsize=(10, 6))

    for data in models_data:
        steps = np.array(data['steps'])
        mean = np.array(data['mean'])
        std = np.array(data['std'])
        model_name = data.get('model_name')
        action_type = data.get('action_type', 'discrete').lower()

        # Choose line style based on action type
        if action_type == 'continuous':
            linestyle = '--'
        elif action_type == 'discrete':
            linestyle = '-'
        else:
            raise ValueError(f"Invalid action_type '{action_type}' for model '{model_name}'.")

        label = f"{model_name} ({action_type.capitalize()})"

        plt.plot(steps, mean, linestyle=linestyle, label=label)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.3)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, title_fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_path, fig_name + ".pdf"), bbox_inches='tight')
    plt.clf()
    return


def load_models_data_from_csv(folder_paths, model_names, model_types):
    """
    Reads CSV files for each model and formats them for plotting learning curves.

    Parameters:
    - folder_path: str, path to the folder containing model CSV files.
    - model_names: list of str, names of the models (filenames are expected to be model_name.csv).
    - model_types: list of str, types of the action space of models.

    Returns:
    - models_data: dict formatted for plot_rl_learning_curves().
    """
    models_data = []

    for path, model, model_type in zip(folder_paths, model_names, model_types):
        mean_path = os.path.join(path, "return_stat-mean.csv")
        std_path = os.path.join(path, "return_stat-std.csv")
        mean_df = pd.read_csv(mean_path)
        std_df = pd.read_csv(std_path)

        models_data.append({
            'steps': mean_df['Step'].values,
            'mean': mean_df['Value'].values,
            'std': std_df['Value'].values,
            'model_name': model,
            "action_type": model_type
        })

    return models_data
