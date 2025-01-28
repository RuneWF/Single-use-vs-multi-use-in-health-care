# Import libaries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import bw2data as bd

# Importing self-made libaries
import standards as s
import life_cycle_assessment as lc
import LCA_plots as lp
import reload_lib as rl
import import_ecoinvent_and_databases as ied

from copy import deepcopy as dc

def get_all_flows(path):
    bd.projects.set_current("single use vs multi use")
    db_type = ['apos', 'consq', 'cut_off']

    flows = {}
    save_dir = {}
    database_name_lst = []
    file_name = {}
    db_type_dct = {}

    for nr in range(1,3):


        # Setting brightway to the given project

        # Specifiyng which database to usey

        for tp in db_type:
            database_name = f'case{nr}' + '_' + tp
            database_name_lst.append(database_name)
            db = bd.Database(database_name)
            # print(database_name)
            flow = []
            if 'case1' in str(db):
                for act in db:
                    temp = act['name']
                    # print(flow)
                    if ('H2' in temp  or 'H4' in temp) and ('SU' in temp or 'REC' in temp) and temp not in flow:
                        flow.append(temp)
                    elif 'alubox' in temp and '+' in temp and 'eol' not in temp.lower():
                        flow.append(temp)
                flow.sort()
            elif 'case2' in str(db):
                for act in db:
                    temp = act['name']
                    if temp == 'SUD' or temp == 'MUD':
                        flow.append(temp)
                flow.sort()
                flow.reverse()
            flows[database_name] = flow
            dir_temp = s.results_folder(path+'\\results', f"case{nr}", tp)
            save_dir[database_name] = dir_temp
            file_name[database_name] = f'{dir_temp}\data_case{nr}_{tp}_recipe.xlsx'
            db_type_dct[database_name] = tp
    
    return flows, database_name_lst, db_type_dct, save_dir, file_name

def initilization(path, lcia_method):

    # ied.database_setup(ecoinevnt_paths, system_path,)

    ui2 = int(input(f'Select 1 to choose flows based on the LCA type, 2 for choosing them yourself and 3 for everything'))

    if ui2 == 1 or ui2 == 2:
        ui1 = int(input('select 1 for case1 and 2 for case2'))
        # Specifying if it is CONSQ (consequential) or APOS
        ui2 = int(input('select 1 for apos and 2 for consequential and 3 for cut off'))
        if ui2 == 1:
            db_type = 'apos'
        elif ui2 == 2:
            db_type = 'consq'
        elif ui2 == 3:
            db_type = 'cut_off'


        # database_project = bw_project
        database_name = f'case{ui1}' + '_' + db_type

        # Creating the flow legend
        if 'case1' in database_name:

            file_identifier = 'case1'

        else:
            file_identifier = 'case2'

        # Creating the saving directory for the results
        save_dir = s.results_folder(path+'\\results', file_identifier, db_type)
        file_name = f'{save_dir}\data_{file_identifier}_{db_type}_{lcia_method}.xlsx'

        if ui2 == 1:
            flows = lc.get_database_type_flows(database_name)
        elif ui2 == 2:
            flows = lc.get_user_specific_flows(database_name)

        return flows, database_name, db_type, save_dir, file_name

    elif ui2 == 3:
        flows, database_name, db_type, save_dir, file_name = get_all_flows(path)
        return flows, database_name, db_type, save_dir, file_name






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
        if 'model' in database_name and 'MUD' in idx:
            min_max_lst = sterilization_min_max(db_type)
            tot_err_min += min_max_lst[0]
            tot_err_max += min_max_lst[1]
            print(min_max_lst)
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
    if 'case2' in database_name.lower():
        plt.title(f"Sensitivity analysis of electrosurgery for total GWP impact for 1 FU - {db_type}", weight='bold')  # Add title
    elif 'case1' in database_name.lower():
        plt.title(f"Sensitivity analysis of sterilization for total GWP impact for 1 FU - {db_type}", weight='bold')
    else:
        plt.title(f"Sensitivity analysis of sterilization for total GWP impact for 1 FU - {db_type}", weight='bold')

    plt.tight_layout()

    # Save and display plot
    plt.savefig(os.path.join(save_dir, f'sensitivity_analysis_{database_name}_{db_type}.jpg'), bbox_inches='tight')
    plt.show()

    return df_tot_err, df_err_min, df_err_max