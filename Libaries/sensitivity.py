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
            # flow_legend = [
            #             'H2S',
            #             'H2R',
            #             'ASC',
            #             'ASW',
            #             'H4S',
            #             'H4R',
            #             'ALC',
            #             'ALW'
            #             ]
            file_identifier = 'case1'

        else:
            # flow_legend = ['SUD', 'MUD']
            file_identifier = 'case2'

        # Specifying the file name and sheet name

        # sheet_name = f'{file_identifier}'

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



def break_even_initialization(path, lcia_method):
    # Reloading the self made libaries to ensure they are up to date
    flows, database_name, db_type, save_dir, file_name = initilization(path, lcia_method)
    # initialization = [database_project, database_name, flows, lcia_method, db_type]
    impact_category = lc.lcia_method(lcia_method)
    if type(flows) is list:
        # Extracting the variables used
        
        

        # Importing the results data frame
        df = lc.import_LCIA_results(file_name, flows, impact_category)

        df_rearranged = lc.rearrange_dataframe_index(df, database=database_name)
        if 'recipe' in lcia_method:
            df_res, df_endpoint = lc.recipe_dataframe_split(df_rearranged)
        else:
            df_res = df_rearranged

        # seperating the GWP potential from the resy
        df_col = [df_res.columns[1]]
        df_GWP = df_res[df_col]

        variables = [database_name, df_GWP, db_type, save_dir, flows, impact_category]
    else:
        variables = {}
        for db, (key, item) in enumerate(flows.items()):
            
            # Importing the results data frame
            df = lc.import_LCIA_results(file_name[key], item, impact_category)

            df_rearranged = lc.rearrange_dataframe_index(df, database_name[db])
            if 'recipe' in lcia_method:
                df_res, df_endpoint = lc.recipe_dataframe_split(df_rearranged)
            else:
                df_res = df_rearranged

            # seperating the GWP potential from the resy
            df_col = [df_res.columns[1]]
            df_GWP = df_res[df_col]

            variables[key] = [database_name[db], df_GWP, db_type[key], save_dir[key], flows[key], impact_category]
    return variables

def sterilization_min_max(database_type):
    impact_category = lc.lcia_method('recipe')

    database_project = 'SU_vs_MU'
    database_name = 'sterilization'
    flow = lc.get_database_type_flows(database_project, database_name, database_type)
    file_name = f'C:\\Users\\ruw\\Desktop\\RA\\Single-use-vs-multi-use-in-health-care\\results\\sterilization_{database_type}\\data_sterilization_{database_type}_recipe.xlsx'
    df = lc.import_LCIA_results(file_name, flow, impact_category)
    df_tot, df_scaled = lc.dataframe_element_scaling(df)
    gwp_col = df_tot.columns[1]
    df_gwp = df_tot[gwp_col].to_frame()
    df_sens = dc(df_gwp)

    min = None
    max = None
    min_idx = ''
    max_idx = ''

    for col in df_sens:
        print(col)
        for idx, row in df_sens.iterrows():
            val = row[col]
            print(idx, val)
            if '200' in idx or 'small' in idx:
                val /= 4
            else:
                val /= 6
            if max is None or min is None:
                min = val
                max = val

            elif val < min:
                min = val
                min_idx = idx
            elif val > max:
                max = val
                max_idx = idx



    # print(min_idx, min, max_idx, max)
    used_val = (df_tot.iloc[0, 0])

    min_max_lst = [used_val/4 - min, max - used_val/4]
    print(min_idx, min, max_idx, max)
    return min_max_lst

def uncertainty_values(df_stack_updated, database_name):
    # Calling the function to have the different activiteis split into the correct column in the dataframe
    df_be = lp.break_even_orginization(df_stack_updated, database_name)

    # Creating two dataframes for the minimum and maximum values
    df_err_min = pd.DataFrame(0, index=df_be.index, columns=df_be.columns, dtype=object)
    df_err_max = pd.DataFrame(0, index=df_be.index, columns=df_be.columns, dtype=object)

    df_err = [df_err_min, df_err_max]

    # Finding the minimimum and maximum value of the sensitivity analysis
    if 'case1' in database_name:

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

    elif 'case2' in database_name:
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
                    # elif 'MUD' in idx:

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
    colors = s.plot_colors(database_name,'turbo')

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