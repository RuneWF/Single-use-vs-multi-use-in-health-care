# Import libaries
import pandas as pd
from copy import deepcopy as dc
import os
from openpyxl import load_workbook


# Importing self-made libaries
import standards as s
import life_cycle_assessment as lc
import LCA_plots as lp
import sensitivity_case1 as c1
import sensitivity_case2 as c2


import bw2data as bd



def column_sum_dataframe(df_sensitivity_v):
    tot_dct = {}
    for col in df_sensitivity_v.columns:
        tot_dct[col] = 0
        for idx, row in df_sensitivity_v.iterrows():
            if idx != 'total':
                tot_dct[col] += row[col]
    return tot_dct

def sensitivity_table_results(totals_df, idx_sens, col_to_df, df_sensitivity_v):
    tot_lst = {}
    for tidx in totals_df.index:
        tot_lst[tidx] = totals_df.at[tidx, 'Value']

    df_sensitivity_p = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    dct = column_sum_dataframe(df_sensitivity_v)

    for col in df_sensitivity_v.columns:
        for idx, row in df_sensitivity_v.iterrows():
            if 'lower' in col:
                tidx = col.replace(" - lower%", "")
                tot = tot_lst[tidx]
            else:
                tidx = col.replace(" - upper%", "")
                tot = tot_lst[tidx]

            if row[col] != 0 and 'total' not in idx:
                val = row[col]
                if 'lower' in col:
                    sens = tot - val
                else:
                    sens = tot + val
                df_sensitivity_p.at[idx, col] = (val - tot)/tot*100
            elif 'total' in idx:
                if 'lower' in col:
                    tidx = col.replace(" - lower%", "")
                    tot = tot_lst[tidx]
                    df_sensitivity_p.at[idx, col] = ((tot - dct[col]) - tot)/tot * 100
                else:
                    tidx = col.replace(" - upper%", "")
                    tot = tot_lst[tidx]
                    df_sensitivity_p.at[idx, col] = ((tot + dct[col]) - tot)/tot * 100
            if round(df_sensitivity_p.at[idx, col],2) != 0:
                df_sensitivity_p.at[idx, col] = f"{df_sensitivity_p.at[idx, col]:.2f}%"
            else:
                df_sensitivity_p.at[idx, col] = "-"

    return df_sensitivity_p

def autoclave_gwp_impact_case1(variables, path):
    database_name, _, db_type, _, flows, impact_category = variables
    db = bd.Database(database_name)
    unique_process_index = []
    for act in db:
        for f in flows:
            for exc in act.exchanges():
                if act['name'] == f and str(exc.input) not in unique_process_index and exc['type'] != 'production':
                    unique_process_index.append(str(exc.input))

    unique_process_index.sort()
    save_dir_case1 = s.results_folder(path+'\\results', 'case1', db_type)
    results_path = os.path.join(save_dir_case1, f"data_uniquie_case1_{db_type}_recipe.xlsx")
    df_unique = lc.import_LCIA_results(results_path, impact_category)
    autoclave_gwp = None
    for location in ["DK", "GLO", "RER"]:
        try:
            autoclave_gwp = df_unique.at[f"'autoclave' (unit, {location}, None)", impact_category[1]]
            if autoclave_gwp is not None:
                break
        except KeyError:
            continue
    if autoclave_gwp is None:
        raise KeyError("Autoclave GWP impact not found for DK, GLO, or RER.")

    return autoclave_gwp

def results_dataframe(df_sensitivity, df_dct, df_be):
    df_perc = dc(df_sensitivity)
    for sc, case in df_dct.items():
        # print(sc, case)
        for desc, data in case.items():
            if desc != 'total':
                
                idx = [i for i in data.index][0]

                lst1 = data.loc[idx].to_list()
                lst2 = df_be.loc[idx].to_list()
                tot_sens = 0
                tot = 0
                for k, n in enumerate(lst1):
                    tot_sens += n
                    tot += lst2[k]
                # print(idx, desc, tot, tot_sens)
                p = (tot_sens - tot)/tot
                df_perc.at[desc, sc] = p

    results_df = dc(df_perc)
    for col in results_df.columns:
        for idx, row in results_df.iterrows():
            if 'total' not in idx:
                results_df.at['total', col] += row[col]
            if row[col] == 0:
                row[col] = '-'

    return results_df 

def calculate_sensitivity_values(variables, autoclave_gwp):
    database_name, df_GWP, db_type, save_dir, impact_category, flows = variables
    if 'case1' in database_name:
            flow_legend = [
                        'H2I',
                        'H2R',
                        'ASC',
                        'ASW',
                        'H4I',
                        'H4R',
                        'ALC',
                        'ALW'
                        ]

    else:
        flow_legend = ['SUD', 'MUD']

    # Creating the dataframe for min and max values
    columns = lc.unique_elements_list(database_name)
    df_stack_updated, totals_df = lp.process_categorizing(df_GWP, db_type, database_name, 'break even', flow_legend, columns)
    # Calling the function to have the different activiteis split into the correct column in the dataframe
    df_be = lp.break_even_orginization(df_stack_updated, database_name)
    # Finding the minimimum and maximum value of the sensitivity analysis
    if 'case1' in database_name:
        df, val_dct, idx_sens, col_to_df = c1.case1_initilazation(df_be)
        df_dct =c1.uncertainty_case1(df, val_dct, df_be, totals_df, idx_sens, col_to_df)
        return results_dataframe(df, df_dct, df_be)
    # c1.uncertainty_case1(df, val_dct, df_be, totals_df, idx_sens, col_to_df), df_be
    
    elif 'case2' in database_name:
        df, val_dct, idx_sens, col_to_df = c2.case2_initilazation(df_be, db_type, autoclave_gwp)
        df_dct = c2.uncertainty_case2(val_dct, df_be, df)
        return results_dataframe(df, df_dct, df_be)

def save_sensitivity_to_excel(variables, path, autoclave_gwp_dct):
    identifier = variables[0]
    save_dir = variables[3]
    df_sens = calculate_sensitivity_values(variables, autoclave_gwp_dct)

    results_path = os.path.join(save_dir, f"sensitivity_{identifier}.xlsx")
    
    if os.path.exists(results_path):
        try:
            # Try to load the existing workbook
            book = load_workbook(results_path)
            with pd.ExcelWriter(results_path, engine='openpyxl', mode='a') as writer:
                writer.book = book
                df_sens.to_excel(writer, sheet_name=identifier, index=True)
        except Exception as e:
            print(f"Error loading existing workbook: {e}")
            # If there's an error loading the workbook, create a new one
            with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
                df_sens.to_excel(writer, sheet_name=identifier, index=True)
    else:
        # If the file does not exist, create a new one
        with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
            df_sens.to_excel(writer, sheet_name=identifier, index=True)
    
    print(f"Saved successfully to {results_path} in sheet {identifier}")

    return df_sens

def obtain_case1_autoclave_gwp(variables, path):
    autoclave_gwp_dct = {}
    for key, item in variables.items():
        try:
            if '1' in key:
                autoclave_gwp_dct[f'case2_{item[2]}'] = autoclave_gwp_impact_case1(item, path)
                autoclave_gwp_dct[item[0]] = ''
        except KeyError as e:
            print(e)
            
    return autoclave_gwp_dct

def iterative_save_sensitivity_results_to_excel(variables, path):
    autoclave_gwp_dct = obtain_case1_autoclave_gwp(variables, path)
    df_dct = {}
    df_dct_be = {}
    for key, item in variables.items():
       
        if '1' in key:
            df_sens = save_sensitivity_to_excel(item, path, autoclave_gwp_dct[key])
        elif '2' in key:
            df_sens = save_sensitivity_to_excel(item, path, autoclave_gwp_dct[key])
        df_dct[key] = df_sens
    return df_dct, df_dct_be

