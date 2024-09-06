import numpy as np
import matplotlib.pyplot as plt

def plot_colors(list_length, color):
    cmap = plt.get_cmap(color)
    colors = [cmap(i) for i in np.linspace(0, 1, len(list_length))]
    return colors

# https://scales.arabpsychology.com/stats/how-do-you-swap-two-rows-in-pandas/
def swap_rows(df, row1, row2):
    df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
    return df






