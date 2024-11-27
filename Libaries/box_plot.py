import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def box_plot(path, inputs, plot_structure):
    c0, c1, c2, plot_label, save_name, leg_size, y_min, y_max, ystep, life_time = plot_structure
    colors = inputs[1]
    save_dir = inputs[2]

    box_plot_df = pd.read_excel(path)
    df_bp = ((box_plot_df['Use pr day']).dropna()).to_frame()

    for col in df_bp.columns:
        for i, row in df_bp.iterrows():
            row[col] *= 365 * life_time

    # Update font size
    plt.rcParams.update({'font.size': 10})

    # Create the boxplot
    boxplot = sns.boxplot(
        data=df_bp, 
        flierprops=dict(marker='o', color='red', markerfacecolor='w', markersize=6), 
        color = colors[c0],
        linewidth=1.0  # Adjust overall line width for whiskers, caps, etc.
    )

    # Access the Q1-Q3 box elements and customize their linewidths
    for box in boxplot.artists:  # `artists` contains the box elements
        box.set_linewidth(0.8)  # Adjust the box thickness

    
    
    # Customize the colors for the plot components
    for patch in boxplot.artists:
        patch.set_facecolor(colors[c0])
        patch.set_edgecolor(colors[c0])
        patch.set_linewidth(3)

    for i, line in enumerate(boxplot.lines):
        # Median lines
        if i % 6 == 4: # Every 6th element starting from 4 (e.g., 4, 10, 16) corresponds to the median line of each boxplot
            line.set_color(colors[c1])
            line.set_linewidth(1)
        # Whiskers
        elif i % 6 == 0 or i % 6 == 1: # Every 6th element starting from 0 or 1 corresponds to the lower and upper whisker lines, respectively
            line.set_color(colors[c0])
            line.set_linewidth(1)
        # Caps
        elif i % 6 == 2 or i % 6 == 3: # Every 6th element starting from 2 or 3 corresponds to the lower and upper caps of the boxplot
            line.set_color(colors[c0])
            line.set_linewidth(1)

    # Add the mean values
    mean_value = df_bp.mean().values[0]  # Since the DataFrame is only one column
    ax = plt.gca()
    ax.scatter(x=0, y=mean_value, color=colors[c2], marker='x', label='Mean',zorder=2)

    # Add the legend with explanations
    legend_elements = [
        plt.Line2D([0], [0], color=colors[c0], lw=6, label='Q1 to Q3'),
        plt.Line2D([0], [0], color=colors[c1], lw=2, label='Median '),
        plt.Line2D([0], [0], color=colors[c2], marker='x', linestyle='None', markersize=6, label='Mean'),
        # plt.Line2D([0], [0], color=colors[c0], markerfacecolor='w', marker='o', linestyle='None', markersize=6, label='Outliers')
    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=((1-leg_size)/len(legend_elements), -0.1, leg_size, 0), loc="lower left", mode="expand", borderaxespad=0,  ncol=4, fontsize=10)

    # Customize the plot (optional)
    plt.ylabel(plot_label, fontsize=10, weight='bold')
    ax.get_xaxis().set_visible(False)
    plt.yticks(np.arange(y_min, y_max + 0.001, step=ystep))
    plt.ylim(y_min-0.001, y_max+0.005)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'boxplot_{save_name}.jpg'), bbox_inches='tight')

    plt.show()