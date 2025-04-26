import matplotlib.pyplot as plt
import numpy as np

# Use the "classic" style for a traditional and serious look
plt.style.use('./paper_style.mpl')

# Data and configuration
models = ['switch-base-8', 'switch-base-64', 'switch-base-128']
methods = ['Finetuned', 'SiDA-MoE', 'MoE-MPMC(Ours)']
n_models = len(models)
n_methods = len(methods)

# Finetuning time data for SST2 and MRPC (in hours)
sst2_time = np.array([
    [3.4, 5.4, 4.1],    # switch-base-8
    [9.2, 11.5, 9.4],   # switch-base-64
    [11.5, 14.2, 12.4]   # switch-base-128
])
mrpc_time = np.array([
    [0.46, 0.45, 0.47],    # switch-base-8
    [3.41, 2.31, 2.4],     # switch-base-64
    [9.5, 13.2, 10.6]      # switch-base-128
])

# Define colors for the three methods in a subtle palette
colors = {
    'Finetuned': '#2E86AB',   # deep blue
    'SiDA-MoE': '#F6C85F',    # mustard yellow
    'MoE-MPMC(Ours)': '#6F4E7C'     # muted purple
}

# Figure and subplots with pure white background
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
fig.patch.set_facecolor('white')  # Set figure background to pure white

# Set axes background to white
for ax in axes:
    ax.set_facecolor('white')

# Settings for grouped bars
bar_width = 0.25
index = np.arange(n_models)

def plot_grouped_bars(ax, data, dataset_name, ylim, y_label):
    # For each method, plot bars with an appropriate offset
    for i, method in enumerate(methods):
        offset = (i - 1) * bar_width  # Center the bars around each group
        bars = ax.bar(index + offset, data[:, i], bar_width,
                      label=method, color=colors[method],
                      edgecolor='black', linewidth=1)
        # Annotate each bar with its value
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    ax.set_xticks(index)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_title(f'{dataset_name} Finetuning Time', fontsize=16, fontweight='bold')
    ax.set_ylim(ylim)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Plot for SST2 finetuning time
plot_grouped_bars(axes[0], sst2_time, 'SST2', ylim=(0, 18), y_label='Time (hours)')

# Plot for MRPC finetuning time
plot_grouped_bars(axes[1], mrpc_time, 'MRPC', ylim=(0, 15), y_label='Time (hours)')

# Position the legend on top center of the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12, frameon=False,
           bbox_to_anchor=(0.5, 1.05))

plt.tight_layout(rect=[0, 0, 1, 0.95])
# Save the plot as a PNG file with 300 dpi
plt.savefig('finetuning_time_plot.png', dpi=300)
