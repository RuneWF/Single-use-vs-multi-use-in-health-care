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
    
    rl.reload_lib(lib)
    flow_legend, database_name, file_path, _, save_dir, initialization, _, db_type = lc.initilization(path, lcia_method)
    impact_category = lc.lcia_method(lcia_method)

    df = lc.import_LCIA_results(file_path, initialization[2], impact_category)

    df_rearranged = lc.rearrange_dataframe_index(df)
    if 'recipe' in lcia_method:
        df_res, df_endpoint = lc.recipe_dataframe_split(df_rearranged)
    else:
        df_res = df_rearranged
    
    
    df_col = [df_res.columns[1]]
    df_GWP = df_res[df_col]

    return database_name, df_GWP, db_type, flow_legend, save_dir

def uncertainty_values(df_stack_updated, database_name):
    df_be = lp.break_even_orginization(df_stack_updated, database_name)
    df_err_min = pd.DataFrame(0, index=df_be.index, columns=df_be.columns, dtype=object)
    df_err_max = pd.DataFrame(0, index=df_be.index, columns=df_be.columns, dtype=object)
    
    df_err = [df_err_min, df_err_max]

    if 'sterilization' in database_name:

        lst_std = [199, 827]


        use_dct = {}
        sheet_dct  = {}
        for idx in df_err_min.index:
            if '2' in idx:
                use_dct[idx] = [12,18]
                sheet_dct[idx] = [63, 71]
            elif '4' in idx:
                use_dct[idx] = [8,9]
                sheet_dct[idx] = [190, 202]
            elif 'S' in idx:
                use_dct[idx] = [9,12]
            else:
                use_dct[idx] = [6,9]


        for sc, df in enumerate(df_err):
            for col in df.columns:
                for idx, row in df.iterrows():
                    if 'H' not in idx and 'Disinfection' not in col and 'Autoclave' not in col:
                        row[col] = (df_be.at[idx, col] * lst_std[sc] / 513)
                    
                    elif 'Autoclave' in col:
                        row[col] = (df_be.at[idx, col] * use_dct[idx][sc] / use_dct[idx][0])

                    elif 'H' in idx and ('Disinfection' not in col  and 'Autoclave' not in col and 'Recycling' not in col):
                        row[col] = (df_be.at[idx, col] * sheet_dct[idx][sc] / sheet_dct[idx][1])/1000
                    else:
                        row[col] = 0

            
    elif 'model' in database_name:
        life_time = [50,500]
        autoclave = [36, 48]
        cabinet_washer = [32, 48]
        use_elec = ((60-4)/60*40 + 500 * 4/60)/1000
        use_time = [((60-2)/60*40 + 500 * 2/60)/1000, ((60-6)/60*40 + 500 * 6/60)/1000]
        for sc, df in enumerate(df_err):
            for col in df.columns:
                for idx, row in df.iterrows():
                    if 'MUD' in idx and 'Disinfection' not in col and 'Autoclave' not in col:
                        row[col] = (df_be.at[idx, col] * life_time[sc] / 250 )
                    elif 'MUD' in idx and 'Autoclave' in col:
                        row[col] = (df_be.at[idx, col] * autoclave[sc] /autoclave[0] )
                    elif 'MUD' in idx and 'Autoclave' in col:
                        row[col] = (df_be.at[idx, col] * cabinet_washer[sc] / cabinet_washer[0])
                    elif 'use' in col.lower():
                        row[col] = (df_be.at[idx, col] * use_time[sc] / use_elec)
                    else:
                        row[col] = 0
    else:
        print("Select either SU_vs_MU -> sterilization or dithermy -> model or create your own sensitivity values ")

    colors_lst = [c for c in df_be.columns]
    return df_err_min, df_err_max, colors_lst

def uncertainty_graph(variables, lib, y_axis):
    rl.reload_lib(lib)

    y_min, y_max, y_step, y_offset  = y_axis
    rl.reload_lib(lib)
    database_name, df_GWP, db_type, flow_legend, save_dir = variables
        
    columns = lc.unique_elements_list(database_name)
    df_stack_updated, totals_df = lp.process_categorizing(df_GWP, db_type, database_name, 'break even', flow_legend, columns)

    df_err_min, df_err_max, colors_lst = uncertainty_values(df_stack_updated, database_name)


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
    colors = s.plot_colors(colors_lst,'turbo')
    for col in tot_df.columns:
        for x_val, (idx, row) in enumerate(tot_df.iterrows()):
            x.append(x_val)
            y.append(row[col])

    y_err = (np.array(y_err_min), np.array(y_err_max))
    _, ax = plt.subplots(figsize=(10, 6))
    plt.errorbar(x,
                 y, 
                 yerr=y_err, 
                 fmt='.', 
                 capsize=6,                  # Add caps
                 ecolor=colors[4],           # Color of error bars
                 markerfacecolor=colors[0],  # Fill color of the marker
                 markeredgecolor=colors[0],  # Outline color of the marker
                 markersize=6,               # Size of the marker
                 elinewidth=2,
                 capthick=2)
                
    
    # plt.legend()  # Add legend to differentiate columns
    ax.set_xticks(x)

    ax.set_xticklabels(plt_index, fontsize=10, weight='bold')
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


