import numpy as np
import matplotlib.pyplot as plt
import os

def plot_colors(uniqie_elements, color):
    cmap = plt.get_cmap(color)
    colors = [cmap(i) for i in np.linspace(0, 1, len(uniqie_elements))]
    return colors

# https://scales.arabpsychology.com/stats/how-do-you-swap-two-rows-in-pandas/
def swap_rows(df, row1, row2):
    df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
    return df

def results_folder(path, name, db):
    save_dir = f'{path}\{name}_{db}'
    temp = f'{name}_{db}'
    try:
        if os.path.exists(save_dir):
             print(f'{temp} already exist')
        else:
            # Create the directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            print(f'The folder {temp} is created')
    
    except UnboundLocalError:
        print('Error occured')
    
    return save_dir






