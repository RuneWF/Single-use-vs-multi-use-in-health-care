import bw2data as bd
import brightway2 as bw 
import bw2calc as bc

import pandas as pd
import copy

# Importing self-made libaries
from standards import *
import LCA_plots as lp
import non_bio_co2 as nbc
import import_ecoinvent_and_databases as ied
from life_cycle_assessment import *



def quick_LCIA_calculator(unique_process_index, func_unit, impact_categories, file_name_unique, sheet_name):
    if type(impact_categories) == tuple:
        impact_categories = [ic for ic in impact_categories]
    df_unique = pd.DataFrame(0, index=unique_process_index, columns=impact_categories, dtype=object)

    print(f'Calculating for {len(impact_categories)} methods and {len(func_unit)} activities : Total calculations {len(impact_categories) * len(func_unit)}' )
    bd.calculation_setups['calc_setup'] = {'inv':func_unit, 'ia': impact_categories}
    mylca = bc.MultiLCA('calc_setup')
    res = mylca.results
    for col, arr in enumerate(res):
        for row, val in enumerate(arr):
            df_unique.iat[col, row] = val

    # Specifying the file name and sheet name
    save_LCIA_results(df_unique, file_name_unique, sheet_name, impact_categories)

def redo_LCIA_unique_process(df_unique, initialization, file_name_unique, sheet_name):
    database_project, database_name, flows, lcia_method, db_type =  initialization
    functional_unit, impact_category, plot_x_axis_all = LCA_initialization(database_project, database_name, flows, lcia_method, db_type)


    # Ensure impact categories is a list
    impact_categories = list(impact_category) if isinstance(impact_category, tuple) else impact_category

    uniquie_process = []
    unique_process_index = []
    # Find matches in functional units and calculate LCA
    for func_dict in functional_unit:
        for FU_key, FU_item in func_dict.items():
            for proc in FU_item.keys():
                if proc not in uniquie_process:
                    uniquie_process.append(proc)
                    unique_process_index.append(f'{proc}')
    
    unique_process_index.sort()


    uniquie_process_ordered= [0] * len(uniquie_process)

    # Extracting all the unique process
    for i, upi in enumerate(unique_process_index):
        for proc in uniquie_process:
            if upi == f'{proc}':
                uniquie_process_ordered[i] = proc
                
    redo_func_unit = []
    process_index = []
    # Asking the user which activities shall be recalculated
    for idx in uniquie_process_ordered:
        user_input = input(f'Do you want to redo the calculation for {idx}? [y/n]') # https://www.w3schools.com/python/python_user_input.asp
        if 'y' in user_input.lower():
            redo_func_unit.append({idx : 1})
            process_index.append(f'{idx}')
        


    df_unique_redone = pd.DataFrame(0, index=process_index, columns=impact_categories, dtype=object)

    print(f'Calculating for {len(impact_categories)} methods and {len(redo_func_unit)} activities : Total calculations {len(impact_categories) * len(redo_func_unit)}' )
    # Performing the LCA calculation and saving the results in a dataframe
    bd.calculation_setups['calc_setup'] = {'inv':redo_func_unit, 'ia': impact_categories}
    mylca = bc.MultiLCA('calc_setup')
    res = mylca.results
    for col, arr in enumerate(res):
        for row, val in enumerate(arr):
            df_unique_redone.iat[col, row] = val


    # Inserting the recalculated results in the orginal dataframe
    df_unique_copy = copy.deepcopy(df_unique)
    for col in impact_categories:
        for i, row in df_unique_redone.iterrows():
            df_unique_copy.at[i, col] = row[col]

    save_LCIA_results(df_unique_copy, file_name_unique, sheet_name, impact_categories)
        
    return df_unique_copy

def lcia_dataframe_handling(file_name, sheet_name, impact_categories, file_name_unique, unique_process_index, initialization, FU, functional_unit, flows):
    
    # Check if file exists
    if os.path.isfile(file_name_unique): # https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
        # Import LCIA results
        try:
            df_unique = import_LCIA_results(file_name_unique, unique_process_index, impact_categories)
            user_input = input("Do you want to redo the calculations for some process? [y/n] (select 'a' if you want to redo eveything, or select 'r' to recalculate based only on the FU)?")
            if 'y' in user_input.lower():
                df_unique_new = redo_LCIA_unique_process(df_unique, initialization, file_name_unique, sheet_name)
            elif 'a' in user_input.lower():
                quick_LCIA_calculator(unique_process_index, FU, impact_categories, file_name_unique, sheet_name)

        except ValueError or KeyError: # Recalculating everything if the saved dataframe does not have the same amount of process as the now
            print("ValueError encountered")
            quick_LCIA_calculator(unique_process_index, FU, impact_categories, file_name_unique, sheet_name)
    else:
        quick_LCIA_calculator(unique_process_index, FU, impact_categories, file_name_unique, sheet_name)

    df_unique = import_LCIA_results(file_name_unique, unique_process_index, impact_categories)

    df = pd.DataFrame(0, index=flows, columns=impact_categories, dtype=object)
    
    if 'n' not in user_input.lower():
        

        for col in impact_categories:
            for i, row in df.iterrows():
                row[col] = []

        for col, impact in enumerate(impact_categories):
            for proc, fu in functional_unit.items():
                for key, item in fu.items():
                    exc = str(key)
                    val = float(item)
                    factor = df_unique.at[exc, impact]
                    impact_value = val * factor
                    # exc = exc.replace("'","")
                    # print(key, proc, val, factor, impact_value, val*factor == impact_value)
                    try:
                        df.at[proc, impact].append([exc, impact_value])
                    except ValueError:
                        try:
                            df.at[proc, impact].append([exc, impact_value])
                            exc = exc.replace(")",")'")
                        except ValueError:
                            print(f'value error for {proc}')
        
            
        save_LCIA_results(df, file_name, sheet_name, impact_categories)
    
    return df, df_unique

def quick_LCIA(initialization, file_name, file_name_unique, sheet_name):
    _, database_name, flows, lcia_method, db_type = initialization
    functional_unit, impact_category = LCA_initialization(database_name, flows, lcia_method)


    # Ensure impact categories is a list
    impact_categories = list(impact_category) if isinstance(impact_category, tuple) else impact_category
    # Loop through each impact category and flow
    
    unique_process_index = []
    uniquie_process = []

    for exc in functional_unit.values():
        for proc in exc.keys():
            if str(proc) not in unique_process_index:
                unique_process_index.append(str(proc))
                uniquie_process.append(proc)
    
    unique_process_index.sort()


    FU = []
    for upi in unique_process_index:
        for proc in uniquie_process:
            if upi == f'{proc}':
                FU.append({proc: 1})

    # user_input = ''


    # obtaing a shortened version of the impact categories for the plots
    plot_x_axis_all = [0] * len(impact_categories)
    for i in range(len(plot_x_axis_all)):
        plot_x_axis_all[i] = impact_categories[i][2]

    df, df_unique = lcia_dataframe_handling(file_name, sheet_name, impact_categories, file_name_unique, unique_process_index, initialization, FU, functional_unit, flows)

    return df, plot_x_axis_all, impact_categories, df_unique
