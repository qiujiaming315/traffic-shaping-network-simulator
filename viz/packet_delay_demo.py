import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick


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


def plot_delay_statistics(x_data, y_data, labels, output_path, fig_name, y_err=None, percent_format=True):
    ax = setup_axis()
    x_values, xlabel = x_data
    y_values, ylabel = y_data
    x_values, y_values = np.array(x_values), np.array(y_values)
    ax.set_ylabel(ylabel, labelpad=10, color='#333333', size=40)
    ax.set_xlabel(xlabel, labelpad=15, color='#333333', size=40)
    if percent_format:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    num_line = len(y_values)
    colors = ["#7FB3D5", "#F7CAC9", "#A2C8B5", "#D9AFD9", "#D3D3D3", "#FFFF99", "#FFD1DC", "#9AD1D4", "#B19CD9",
              "#B0AFAF"]
    assert len(colors) >= num_line, "Too many lines to visualize."
    if y_err is None:
        for y_value, color, label in zip(y_values, colors, labels):
            ax.plot(x_values, y_value, 'o-', color=color, label=label, linewidth=3, markersize=9)
    else:
        for y_value, err, color, label in zip(y_values, y_err, colors, labels):
            ax.errorbar(x_values, y_value, err, fmt='o-', color=color, label=label, linewidth=3, markersize=9,
                        ecolor=color, elinewidth=3, capsize=6, capthick=3)
    plt.legend(fontsize=35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.clf()
    return


def plot_delay_distribution(end_to_end_delay, output_path, fig_name):
    ax = setup_axis()
    ax.set_ylabel("Frequency", labelpad=10, color='#333333', size=40)
    ax.set_xlabel("Normalized Packet End-to-end Delay", labelpad=15, color='#333333', size=40)
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.hist(end_to_end_delay, 100, color="blue", weights=np.ones_like(end_to_end_delay) / end_to_end_delay.size,
            label="distribution")
    violation = np.sum(end_to_end_delay > 1) / len(end_to_end_delay)
    ax.axvline(end_to_end_delay.mean(), color="#00CC00", alpha=0.5, linestyle="dashed", linewidth=4, label="average")
    sorted_delay = np.sort(end_to_end_delay)
    ax.axvline(sorted_delay[int(0.5 * len(sorted_delay))], color="#00CC00", alpha=0.5, linewidth=4, label="median")
    ax.axvline(sorted_delay[int(0.95 * len(sorted_delay))], color="#CCCC00", alpha=0.5, linewidth=4,
               label="95th percentile")
    ax.axvline(sorted_delay[int(0.99 * len(sorted_delay))], color="#CC6600", alpha=0.5, linewidth=4,
               label="99th percentile")
    ax.axvline(sorted_delay[-1], color="#CC0000", alpha=0.5, linewidth=4, label="worst case")
    title = "No delay violation" if violation == 0 else f"{violation * 100: .1f} % delay violation"
    ax.set_title(title, color='#333333', pad=20, size=55)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.clf()
    return


def plot_statistics_bar(data, names, ylabel, output_path, fig_name, percent_format=True):
    ax = setup_axis()
    if percent_format:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.bar(np.arange(len(data)), data, width=0.5, color='blue')
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(names, size=20)
    ax.set_ylabel(ylabel, labelpad=10, color='#333333', size=40)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.clf()
    return
