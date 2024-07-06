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


def plot_delay(arrival_time, end_to_end_delay, output_path, fig_name, segment=None):
    ax = setup_axis()
    ax.set_ylabel("End-to-end Delay", labelpad=10, color='#333333', size=40)
    ax.set_xlabel("Packet Arrival Time", labelpad=15, color='#333333', size=40)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    num_flow = len(arrival_time)
    colors = ["#7FB3D5", "#F7CAC9", "#A2C8B5", "#D9AFD9", "#D3D3D3", "#FFFF99", "#FFD1DC", "#9AD1D4", "#B19CD9",
              "#B0AFAF"]
    assert len(colors) >= num_flow, "Too many flows."
    labels = [f"flow {i + 1}" for i in range(num_flow)]
    arrival_aggregate = []
    for xdata, ydata, color, label in zip(arrival_time, end_to_end_delay, colors, labels):
        xdata, ydata = np.array(xdata), np.array(ydata)
        if segment is not None:
            mask = np.logical_and(xdata >= segment[0], xdata <= segment[1])
            xdata, ydata = xdata[mask], ydata[mask]
        ax.plot(xdata, ydata, 'o-', color=color, label=label, linewidth=3, markersize=9)
        arrival_aggregate.extend(list(xdata))
    ax.hlines(100, np.amin(arrival_aggregate), np.amax(arrival_aggregate), colors="red", linewidth=5,
              label='hard delay bound')
    plt.legend(fontsize=35)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, fig_name + ".png"), bbox_inches='tight')
    plt.clf()


def plot_delay_distribution(end_to_end_delay, output_path, fig_name):
    ax = setup_axis()
    ax.set_ylabel("Frequency", labelpad=10, color='#333333', size=40)
    ax.set_xlabel("Normalized Packet End-to-end Delay", labelpad=15, color='#333333', size=40)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.hist(end_to_end_delay, 100, color="blue", weights=np.ones_like(end_to_end_delay) / end_to_end_delay.size,
            label="distribution")
    violation = np.sum(end_to_end_delay > 100) / len(end_to_end_delay)
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
