# Import libaries
import re
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import os

# Importing self-made libaries
import standards as s
import life_cycle_assessment as lc
import LCA_plots as lp
import reload_lib as rl


def break_even_initialization(path, lcia_method, lib):
    # Reloading the self made libaries to ensure they are up to date
    rl.reload_lib(lib)

    # Extracting the variables used
    flow_legend, database_name, file_path, _, save_dir, initialization, _, db_type = lc.initilization(path, lcia_method)
    impact_category = lc.lcia_method(lcia_method)

    # Importing the results data frame
    df = lc.import_LCIA_results(file_path, initialization[2], impact_category)

    df_rearranged = lc.rearrange_dataframe_index(df)
    if 'recipe' in lcia_method:
        df_res, df_endpoint = lc.recipe_dataframe_split(df_rearranged)
    else:
        df_res = df_rearranged
    
    # seperating the GWP potential from the resy
    df_col = [df_res.columns[1]]
    df_GWP = df_res[df_col]

    variables = [database_name, df_GWP, db_type, flow_legend, save_dir]

    return variables

def uncertainty_values(df_stack_updated, database_name):
    # Calling the function to have the different activiteis split into the correct column in the dataframe
    df_be = lp.break_even_orginization(df_stack_updated, database_name)

    # Creating two dataframes for the minimum and maximum values
    df_err_min = pd.DataFrame(0, index=df_be.index, columns=df_be.columns, dtype=object)
    df_err_max = pd.DataFrame(0, index=df_be.index, columns=df_be.columns, dtype=object)
    
    df_err = [df_err_min, df_err_max]

    # Finding the minimimum and maximum value of the sensitivity analysis
    if 'sterilization' in database_name:

        # Minimum and maximum lifetime use of the alu container
        lst_std = [199, 827]

        use_dct = {}
        sheet_dct  = {}

        for idx in df_err_min.index:
            if '2' in idx:
                use_dct[idx] = [12,18]      # Containers in the autoclave
                sheet_dct[idx] = [63/1000, 71/1000]   # Weight of the sheet without and with the protection layer
            elif '4' in idx:
                use_dct[idx] = [8,9]        # Containers in the autoclave
                sheet_dct[idx] = [190/1000, 202/1000] # Weight of the sheet without and with the protection layer
            elif 'S' in idx:
                use_dct[idx] = [9,12]       # Containers in the autoclave
            else:
                use_dct[idx] = [6,9]        # # Containers in the autoclave

        # Performing the minmum and maximum calculation to extract the values
        for sc, df in enumerate(df_err):
            for col in df.columns:
                for idx, row in df.iterrows():
                    # Finding the min and max values then varying the lifetime of the container
                    if 'H' not in idx and 'Disinfection' not in col and 'Autoclave' not in col:
                        temp = (df_be.at[idx, col] * lst_std[sc] / 513)
                    
                    # Finding the min and max values then varying the quantity of container in the autoclave
                    elif 'Autoclave' in col:
                        temp = (df_be.at[idx, col] * use_dct[idx][sc] / use_dct[idx][0])

                    # Finding the min and max values then varying without and with the protection layer for the sheet
                    elif 'H' in idx and ('Disinfection' not in col  and 'Autoclave' not in col and 'Recycling' not in col):
                        temp = (df_be.at[idx, col] * sheet_dct[idx][sc] / sheet_dct[idx][1])
                    # If none of above criterias is fulfil its set to 0
                    else:
                        temp = 0
                    
                    if sc == 0 and temp != 0:
                        row[col] = df_be.at[idx, col] - temp
                    elif sc == 1 and temp != 0:
                        row[col] = temp - df_be.at[idx, col]
                    else:
                        row[col] = 0
            
    elif 'model' in database_name:
        life_time = [50,500]
        autoclave = [36, 48]
        cabinet_washer = [32, 48]

        # Electricity in the usephase
        use_elec = ((60-4)/60*40 + 500 * 4/60)/1000
        use_elec_var = [((60-2)/60*40 + 500 * 2/60)/1000, ((60-10)/60*40 + 500 * 10/60)/1000]

        # Performing the minmum and maximum calculation to extract the values
        for sc, df in enumerate(df_err):
            for col in df.columns:
                for idx, row in df.iterrows():
                    # Finding the min and max values then varying the lifetime of the bipolar burner
                    if 'MUD' in idx and 'Disinfection' not in col and 'Autoclave' not in col:
                        temp = (df_be.at[idx, col] * life_time[sc] / 250 )
                    # Finding the min and max values then varying the quantity of bipolar burner in the autoclave   
                    elif 'MUD' in idx and 'Autoclave' in col:
                        temp = (df_be.at[idx, col] * autoclave[sc] /autoclave[0])
                    # Finding the min and max values then varying the quantity of bipolar burner in the cabinet washer
                    elif 'MUD' in idx and 'dis' in col.lower():
                        temp = (df_be.at[idx, col] * cabinet_washer[sc] / cabinet_washer[0])
                    # Finding the min and max values then varying the time in use
                    elif 'use' in col.lower():
                        temp = (df_be.at[idx, col] * use_elec_var[sc] / use_elec)
                    # If none of above criterias is fulfil its set to 0
                    else:
                        temp = 0

                    if sc == 0 and temp != 0:
                        row[col] = df_be.at[idx, col] - temp
                    elif sc == 1 and temp != 0:
                        row[col] = temp - df_be.at[idx, col]
                    else:
                        row[col] = 0
                    
    else:
        print("Select either SU_vs_MU -> sterilization or dithermy -> model or create your own sensitivity values ")

    # Obtaining the colors
    colors_lst = [c for c in df_be.columns]
    colors = s.plot_colors(colors_lst,'turbo')

    return df_err_min, df_err_max, colors

def uncertainty_graph(variables, lib, y_axis):
    rl.reload_lib(lib)
    # Extracting values for the plot and to create the plot
    y_min, y_max, y_step, y_offset  = y_axis
    database_name, df_GWP, db_type, flow_legend, save_dir = variables
    
    # Creating the dataframe for min and max values
    columns = lc.unique_elements_list(database_name)
    df_stack_updated, totals_df = lp.process_categorizing(df_GWP, db_type, database_name, 'break even', flow_legend, columns)
    df_err_min, df_err_max, colors = uncertainty_values(df_stack_updated, database_name)

    # Combinging the min and max values into a single dataframe
    tot_err_dict = {}
    for idx, row in df_err_min.iterrows():
        tot_err_min = 0
        tot_err_max = 0
        for col in df_err_min.columns:
            tot_err_min += row[col]
            tot_err_max += df_err_max.at[idx, col]
        tot_err_dict[idx] = [tot_err_min, tot_err_max]

    df_tot_err = pd.DataFrame(tot_err_dict.values(), index=df_err_min.index, columns=['Min', 'Max'])

    
    x = []
    y = []
    y_err_min = df_tot_err['Min'].to_list()
    y_err_max = df_tot_err['Max'].to_list()

    tot_df = totals_df['Value'].to_frame()
    tot_df.index = df_tot_err.index
    plt_index = [i for i in tot_df.index]
    
    # extracting the x and y values for the plot
    for col in tot_df.columns:
        for x_val, (idx, row) in enumerate(tot_df.iterrows()):
            x.append(x_val)
            y.append(row[col])
    # Creating the y error values
    y_err = (np.array(y_err_min), np.array(y_err_max)) # https://stackoverflow.com/questions/31081568/python-plotting-error-bars-with-different-values-above-and-below-the-point
    
    # Creating the plot
    _, ax = plt.subplots(figsize=(10, 6))
    plt.errorbar(x,
                 y, 
                 yerr=y_err, 
                 fmt='.',                    # Defining the marker style
                 capsize=6,                  # Add caps
                 ecolor=colors[4],           # Color of error bars
                 markerfacecolor=colors[0],  # Fill color of the marker
                 markeredgecolor=colors[0],  # Outline color of the marker
                 markersize=6,               # Size of the marker
                 elinewidth=2,               # Thickness of the line
                 capthick=2)                 # Thickness of the caps

    # Adjusting the plot to show relevant information
    ax.set_xticks(x)
    ax.set_xticklabels(plt_index, fontsize=10, weight='bold')
    ax.set_xlim(min(x)-0.8, max(x)+0.8)
    ax.axhline(y = 0, color = 'k', linestyle = '-', zorder=0, linewidth=0.5) # https://matplotlib.org/stable/gallery/misc/zorder_demo.html
    plt.yticks(np.arange(y_min, y_max + y_offset, step=y_step))
    plt.ylim(y_min, y_max + y_offset)
    plt.ylabel("Global Warming Potential [kg CO$_2$e/FU]", fontsize=10, weight='bold')  # Label for y-axis
    if 'model' in database_name.lower():
        plt.title(f"Sensitivity analysis of electrosurgery for total GWP impact for 1 FU - {db_type}", weight='bold')  # Add title
    elif 'steril' in database_name.lower():
        plt.title(f"Sensitivity analysis of sterilization for total GWP impact for 1 FU - {db_type}", weight='bold')
    else:
        plt.title(f"Sensitivity analysis of sterilization for total GWP impact for 1 FU - {db_type}", weight='bold')
    
    plt.tight_layout()

    # Save and display plot
    plt.savefig(os.path.join(save_dir, f'sensitivity_analysis_{database_name}_{db_type}.jpg'), bbox_inches='tight')
    plt.show()  
    
