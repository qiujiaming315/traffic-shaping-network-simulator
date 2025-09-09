import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
import pandas as pd


def setup_axis():
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams['figure.figsize'] = (15, 10)
    ax = plt.subplot()
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.spines['top'].set_color('#606060')
    ax.spines['bottom'].set_color('#606060')
    ax.spines['left'].set_color('#606060')
    ax.spines['right'].set_color('#606060')
    ax.grid(True, color='#bfbfbf', linewidth=1)
    return ax


def plot_rl_learning_curves(models_data, output_path, fig_name, xlabel="Training Steps", ylabel="Average Reward",
                            yrange=None):
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

    ax = setup_axis()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#e377c2"]
    assert len(models_data) <= len(colors)

    for data, color in zip(models_data, colors):
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

        ax.plot(steps, mean, linestyle=linestyle, label=label, color=color, linewidth=3)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.3)

    ax.set_xlabel(xlabel, labelpad=15, color='#333333', size=40)
    ax.set_ylabel(ylabel, labelpad=10, color='#333333', size=40)
    ax.legend(fontsize=35, title_fontsize=45)
    if yrange is not None:
        ax.set_ylim(*yrange)
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


def plot_curve_with_elliptical_errorbars(curve_data, scatter_data, output_path, fig_name, curve_label, scatter_labels,
                                         curve_color=None, percent_format=True):
    """
    Plots a curve with point annotations, error ellipses, and additional scatter points with their own ellipses.

    Parameters:
    - curve_data: dict with keys 'x', 'y', 'xerr', 'yerr', 'annotations' (lists of same length)
    - scatter_data: list of dicts, each with keys 'x', 'y', 'xerr', 'yerr', 'label'
    - curve_label: label for the curve (used in legend)
    - scatter_labels: list of labels for the individual scatter points (optional, can override per-point label)
    - curve_color: color for the curve and its annotations
    - scatter_colors: list of colors for the scatter points
    - ellipse_alpha: alpha for the error ellipse fill
    """
    ax = setup_axis()

    # Plot curve with error ellipses
    x, y = np.array(curve_data['x']), np.array(curve_data['y'])
    xerr, yerr = np.array(curve_data['xerr']), np.array(curve_data['yerr'])
    if percent_format:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        y *= 100
        yerr *= 100
    annotations = curve_data.get('annotations', ["" for _ in x])
    curve_color = "#4b0082" if not curve_color else curve_color
    scatter_colors = ["#006466", "#1f77b4", "#ff7f0e", "#2ca02c", "#e377c2"]

    ax.plot(x, y, marker='o', color=curve_color, label=curve_label, linewidth=5, markersize=15, zorder=2)
    for xi, yi, xe, ye, text in zip(x, y, xerr, yerr, annotations):
        ellipse = Ellipse((xi, yi), width=2 * xe, height=2 * ye, alpha=0.2,
                          facecolor=curve_color, edgecolor='none', zorder=1)
        ax.add_patch(ellipse)
        ax.text(xi, yi, text, fontsize=20, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), zorder=3)
        # ax.text(xi, yi, text, fontsize=24, ha='center', va='center', zorder=3)

    # Plot scatter points with error ellipses
    if scatter_labels is None:
        scatter_labels = [d.get('label', f"Point {i}") for i, d in enumerate(scatter_data)]

    for i, (point, color, label) in enumerate(zip(scatter_data, scatter_colors, scatter_labels)):
        xi, yi = point['x'], point['y']
        xe, ye = point['xerr'], point['yerr']
        if percent_format:
            yi *= 100
            ye *= 100
        ax.scatter(xi, yi, color=color, edgecolor='none', label=label, s=250, zorder=4)
        ellipse = Ellipse((xi, yi), width=2 * xe, height=2 * ye, alpha=0.2,
                          facecolor=color, edgecolor='none', zorder=1)
        ax.add_patch(ellipse)

    # Axis formatting
    ax.set_xlabel("Average Delay", labelpad=15, color='#333333', size=40)
    ax.set_ylabel("Delay Violation Percentage", labelpad=10, color='#333333', size=40)
    if len(scatter_labels) > 0:
        ax.legend(fontsize=35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_path, fig_name + ".pdf"), bbox_inches='tight')
    plt.clf()
    return


def plot_curve_with_log_y_and_highlighted_regions(x, y, background_regions, vertical_lines, inset_xlim, inset_ylim,
                                                  output_path, fig_name, curve_label, region_label, line_label,
                                                  inset_bbox=[0.35, 0.35, 0.3, 0.25]):
    """
    Plots a curve with log-scale y-axis and optional background highlight regions.

    Parameters:
    - x, y: Arrays or lists of x and y values for the curve.
    - background_regions: List of (start, end) tuples for x-axis highlight spans.
    - curve_label: Label for the main curve.
    - curve_color: Color for the main curve.
    - region_labels: List of labels for each region to appear in legend.
    - region_colors: List of colors for each background region.
    - region_alpha: Transparency level of background region fill.
    - xlabel, ylabel: Axis labels.
    - log_base: Base of the logarithm for the y-axis (default is 10).
    """
    ax = setup_axis()
    curve_color = "#ff7f0e"
    region_color = "#ffc107"
    # Plot the main curve
    ax.plot(x, y, marker='o', color=curve_color, label=curve_label, linewidth=5, markersize=15)

    # Set log scale for y-axis
    ax.set_yscale('log', base=10)

    # Highlight regions
    for (start, end) in background_regions:
        ax.axvspan(start, end, color=region_color, alpha=0.2)
    region_handle = [Patch(facecolor=region_color, edgecolor='none', alpha=0.2, label=region_label)]

    # Vertical lines
    for i, vline_x in enumerate(vertical_lines):
        ax.axvline(x=vline_x, color="#9c27b0", linestyle="--", linewidth=3)
    vline_proxy = Line2D([0], [0], color="#9c27b0", linestyle="--", linewidth=3, label=line_label)
    region_handle.append(vline_proxy)

    # Inset plot
    # x1, x2, y1, y2 = 0.0035, 0.0065, -0.1, 0.1  # subregion of the original image
    axins = ax.inset_axes(inset_bbox, xlim=inset_xlim, ylim=inset_ylim, xticklabels=[], yticklabels=[])
    axins.plot(x, y, marker='o', color=curve_color, linewidth=5, markersize=15)
    for (start, end) in background_regions:
        axins.axvspan(start, end, color=region_color, alpha=0.2)
    for i, vline_x in enumerate(vertical_lines):
        axins.axvline(x=vline_x, color="#9c27b0", linestyle="--", linewidth=3)
    axins.tick_params(axis='x', labelsize=20)
    axins.tick_params(axis='y', labelsize=20)
    axins.spines['top'].set_color('#606060')
    axins.spines['bottom'].set_color('#606060')
    axins.spines['left'].set_color('#606060')
    axins.spines['right'].set_color('#606060')
    # axins.grid(True, color='#bfbfbf', linewidth=1)
    ax.indicate_inset_zoom(axins, edgecolor="#606060")

    # Labels and legend
    ax.set_xlabel("Simulation Time (seconds)", labelpad=15, color='#333333', size=40)
    ax.set_ylabel(r"Control Parameter $(w)$", labelpad=10, color='#333333', size=40)

    # Combine curve and region labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    handles += region_handle
    labels += [region_label, line_label]
    ax.legend(handles, labels, fontsize=35, loc="center right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_path, fig_name + ".pdf"), bbox_inches='tight')
    plt.clf()
    return
