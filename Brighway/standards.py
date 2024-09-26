def import_libaries():
    # Import packages we'll need later on in this tutorial
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.ticker as mtick
    import matplotlib.cm as cm
    import math
    import plotly.graph_objects as go
    from collections import OrderedDict
    from matplotlib.lines import Line2D  # Import for creating custom legend markers
    import json
    import copy
    import random
    import re
    import seaborn as sns
    import importlib


    # Import BW25 packages
    import bw2data as bd
    import bw2io as bi
    import bw2calc as bc
    import bw2analyzer as bwa
    import brightway2 as bw 
    from bw2calc import LeastSquaresLCA

import_libaries()

def plot_colors(list_length, color):
    cmap = plt.get_cmap(color)
    colors = [cmap(i) for i in np.linspace(0, 1, len(list_length))]
    return colors

# https://scales.arabpsychology.com/stats/how-do-you-swap-two-rows-in-pandas/
def swap_rows(df, row1, row2):
    df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
    return df

def results_folder(name):
    save_dir = f'Single-use-vs-multi-use-in-health-care\{name}'
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    print(f'Folder name {name} created')
    return save_dir





