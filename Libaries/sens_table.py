# Import libaries
import pandas as pd


# Importing self-made libaries
import standards as s
import life_cycle_assessment as lc
import LCA_plots as lp

import bw2data as bd

import os

from openpyxl import load_workbook

from copy import deepcopy as dc

# def initilization(db_type, path, bw_project="single use vs multi use"):
#     database_name = f'case1' + '_' + db_type
#     flows = lc.get_database_type_flows(database_name)
    

#     return bw_project, database_name, flows, save_dir_case1

def LCA_initialization(database_name: str, flows: list, method: str) -> tuple:
    # all_acts, eidb = database_initialization(db_type, database_name, project_name)

    # Setting up an empty dictionary with the flows as the key
    procces_keys = {key: None for key in flows}
    size = len(flows)
    db = bd.Database(database_name)
    for act in db:
            for proc in range(size):
                if act['name'] == flows[proc]:
                    procces_keys[act['name']] = act['code']

    process = []
    key_counter = 0

    # Obtaining all the subprocess in a list
    for key, item in procces_keys.items():
        try:
            process.append(db.get(item))
        except KeyError:
            print(f"Process with key '{item}' not found in the database '{db}'")
            process = None
        key_counter += 1

    # Obtaing the impact categories for the LCIA calculations
    impact_category = lc.lcia_method(method)

    product_details = {}
    product_details_code = {}

    # Obtaining the subsubprocess'
    if process:
        for proc in process:
            product_details[proc['name']] = []
            product_details_code[proc['name']] = []

            for exc in proc.exchanges():
                if 'Use' in exc.output['name'] and exc['type'] == 'biosphere':
                    product_details[proc['name']].append({exc.input['name']: [exc['amount'], exc.input]})
                    # if exc.input in eidb or exc.input in eidb_db or exc.input in eidb_cyl:
                    #     product_details_code[proc['name']].append([exc.output, exc.output['name'], exc.output['code'], exc['amount']])
                elif exc['type'] == 'technosphere':
                    product_details[proc['name']].append({exc.input['name']: [exc['amount'], exc.input]})
                    # if exc.input in eidb or exc.input in eidb_db or exc.input in eidb_cyl:
                    #     product_details_code[proc['name']].append([exc.input, exc.input['name'], exc.input['code'], exc['amount']])



    FU = {key: {} for key in product_details.keys()}
    # Creating the FU to calculate for
    for key, item in product_details.items():
        for idx in item:
            for n, m in idx.items():
                FU[key].update({m[1]: m[0]})

    return FU, impact_category

def process_row(idx, row, col, val_dct, df_be):
            for key, dct in val_dct.items():
                if idx in key:
                    for c in df_be.columns:
                        for i in df_be.index:
                            temp = calculate_temp_tot(idx, col, dct, df_be, i, c)
                            if temp != 0:
                                row[col] = temp

def calculate_temp_tot(idx, col, dct, df_be, i, c):
    temp = 0
    for sc, lst in dct.items():
        if sc in col:
            val = lst[0] if 'lower' in col else lst[1]
            temp += calculate_temp(idx, col, df_be, i, c, val, lst)
    return temp

def calculate_temp(idx, col, df_be, i, c, val, lst):
    if idx == 'Life time' and i in col and 'H' not in i and 'Disinfection' not in c and 'Autoclave' not in c:
        return df_be.at[i, c] * val / 513
    elif idx == 'autoclave' and 'Autoclave' in c:
        return calculate_autoclave_temp(col, df_be, i, c, val)
    elif idx == 'protection cover' and 'H' in i and 'Disinfection' not in col and 'Autoclave' not in col and 'Recycling' not in col:
        return df_be.at[i, c] * val / lst[1]
    return 0

def calculate_autoclave_temp(col, df_be, i, c, val):
    if '2' in col:
        return df_be.at[i, c] * val / 12
    elif '4' in col:
        return df_be.at[i, c] * val / 8
    elif 'AS' in col:
        return df_be.at[i, c] * val / 9
    elif 'AL' in col:
        return df_be.at[i, c] * val / 6
    return 0

def column_sum_dataframe(df_sensitivity_v):
    tot_dct = {}
    for col in df_sensitivity_v.columns:
        tot_dct[col] = 0
        for idx, row in df_sensitivity_v.iterrows():
            if idx != 'total':
                tot_dct[col] += row[col]
    return tot_dct

def sensitivity_table_results(totals_df, idx_sens, col_to_df, df_sensitivity_v):
    # tot_dct = column_sum_dataframe(df_sensitivity_v)
    tot_lst = []
    for tidx in totals_df.index:
        tot_lst.append(totals_df.at[tidx, 'Value'])

    df_sensitivity_p = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    dct = column_sum_dataframe(df_sensitivity_v)

    for col in df_sensitivity_v.columns:
        for idx, row in df_sensitivity_v.iterrows():
            
            if row[col] != 0 and 'total' not in idx:
                val = row[col]

                if 'lower' in col:
                    tidx = col.replace(" - lower%", "")
                    tot = totals_df.at[tidx, 'Value']
                    
                    sens = tot - val
                else:
                    tidx = col.replace(" - upper%", "")
                    tot = totals_df.at[tidx, 'Value']
                    sens = tot + val

                df_sensitivity_p.at[idx, col] = (sens - tot)/tot*100
            elif 'total' in idx:
                if 'lower' in col:
                    df_sensitivity_p.at[idx, col] = ((tot - dct[col]) - tot)/tot * 100
                else:
                    df_sensitivity_p.at[idx, col] = ((tot + dct[col]) - tot)/tot * 100
            if df_sensitivity_p.at[idx, col] != 0:
                df_sensitivity_p.at[idx, col] = f"{df_sensitivity_p.at[idx, col]:.2f}%"
            else:
                df_sensitivity_p.at[idx, col] = "-"

    return df_sensitivity_p

def uncertainty_case1(df_sensitivity, val_dct, df_be, totals_df, idx_sens, col_to_df):
    """
    Perform sensitivity analysis for case 1.

    Parameters:
    df_sensitivity (pd.DataFrame): DataFrame to store sensitivity results.
    val_dct (dict): Dictionary containing sensitivity values.
    df_be (pd.DataFrame): DataFrame containing break-even analysis data.

    Returns:
    pd.DataFrame: Updated DataFrame with sensitivity analysis results.
    """
    

    for col in df_sensitivity.columns:
                for idx, row in df_sensitivity.iterrows():
                    process_row(idx, row, col, val_dct, df_be)

    idx_col = totals_df.columns[0]
    idx_tot = [i[0] for i in totals_df[idx_col]]
    totals_df.index = idx_tot

    

    return sensitivity_table_results(totals_df, idx_sens, col_to_df, df_sensitivity)

def uncertainty_case2(val_dct, df_be, totals_df, idx_sens, col_to_df):
    use_elec = ((60-4)/60*40 + 500 * 4/60)/1000
    df_sensitivity_v = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    idx_col = totals_df.columns[0]
    idx_tot = [i[0] for i in totals_df[idx_col]]
    totals_df.index = idx_tot

    tot_lst = []
    for tidx in totals_df.index:
        tot_lst.append(totals_df.at[tidx, 'Value'])

    for col in df_sensitivity_v.columns:
        for idx, row in df_sensitivity_v.iterrows():
            for key, dct in val_dct.items():
                    if idx in key:
                        for i, r in df_be.iterrows():
                            for c in df_be.columns:
                                for sc, lst in dct.items():
                                    temp = 0
                                    tot_baseline = 0
                                    if sc in i:
                                        tot_baseline += df_be.at[i, c]
                                        if 'lower' in col:
                                            val = lst[0]
                                            tidx = col.replace(" - lower%", "")
                                        else:
                                            val = lst[1]
                                            tidx = col.replace(" - upper%", "")
                                        if idx == 'Life time' and i in col and 'SUD' not in col and 'Disinfection' not in c and 'autoclave' not in c:                                        
                                            temp += (df_be.at[i, c] * val / 250)
                                            
                                        elif idx == 'autoclave' and 'autoclave' in c and 'SUD' not in col:
                                            temp += (df_be.at[i, c] * val / (12*4))
                                        elif idx == 'sterilization' and 'consumables' in c and 'SUD' not in col:
                                            temp += val
                                        elif idx == 'cabinet washer' and 'Disinfection' in c and 'SUD' not in col:
                                            temp += (df_be.at[i, c] * val / 32)
                                            
                                        elif idx == 'surgery time' and 'Disinfection' in c:
                                            temp += (df_be.at[i, c] * val / use_elec)
                                    if temp != 0:
                                        row[col] = temp
    
                
    

    return sensitivity_table_results(totals_df, idx_sens, col_to_df, df_sensitivity_v)

def case1_initilazation(df_be):
    idx_sens = [
            'Life time',
            'autoclave',
            'protection cover',
            'total'
        ]


    val_dct = {
        'Life time' : {},
        'autoclave' : {},
        'protection cover' : {}
    }

    col_to_df = []

    for idx in df_be.index:
        if '2' in idx:
            val_dct['autoclave'].update({idx : [12,18]})
            val_dct['protection cover'].update({idx : [63/1000, 71/1000]})
        elif '4' in idx:
            val_dct['autoclave'].update({idx : [8,9]})
            val_dct['protection cover'].update({idx : [190/1000, 202/1000]})
        elif 'S' in idx:
            val_dct['Life time'].update({idx : [314, 827]})
            val_dct['autoclave'].update({idx : [9,12]})
        else:
            val_dct['Life time'].update({idx : [314, 827]})
            val_dct['autoclave'].update({idx : [6,9]})

        col_to_df.append(f'{idx} - lower%')
        col_to_df.append(f'{idx} - upper%')

    df = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    return df, val_dct, idx_sens, col_to_df

def autoclave_gwp_impact(variables, path):
    database_name, df_GWP, db_type, save_dir, flows, impact_category = variables
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
    df_unique = lc.import_LCIA_results(results_path, unique_process_index, impact_category)
    autoclave_gwp = df_unique.at["'autoclave' (unit, GLO, None)", impact_category[1]]

    return autoclave_gwp

def sterilization_min_max(database_type, autoclave_gwp):
    impact_category = lc.lcia_method('recipe')
    # df_unique = autoclave_gwp_impact(save_dir, database_type)


    database_name = f'case1_{database_type}'
    flow = lc.get_database_type_flows(database_name)
    file_name = f'C:\\Users\\ruw\\Desktop\\RA\\Single-use-vs-multi-use-in-health-care\\results\\case1_{database_type}\\data_case1_{database_type}_recipe.xlsx'
    df = lc.import_LCIA_results(file_name, flow, impact_category)
    df_tot, df_scaled = lc.dataframe_element_scaling(df)
    gwp_col = df_tot.columns[1]
    df_gwp = df_tot[gwp_col].to_frame()
    df_sens = dc(df_gwp)

    min = None
    max = None
    min_auto = 0

    max_auto = 0
    for col in df_sens:
        for idx, row in df_sens.iterrows():
            val = row[col]
            if '200' in idx or 'small' in idx:
                val /= 4
            else:
                val /= 6
            if max is None or min is None:
                min = val
                max = val

            elif val < min:
                min = val
                if '2' in idx:
                    min_auto = 12 * 4
                elif '4' in idx:
                    min_auto = 8 *6
                elif 'S' in idx:
                    min_auto = 9 * 4
                else:
                    min_auto = 6 * 6
                
            elif val > max:
                if '2' in idx:
                    max_auto = 18 *4
                elif '4' in idx:
                    max_auto = 9 * 6
                elif 'S' in idx:
                    max_auto = 12 * 4
                else:
                    max_auto = 9 * 6
                max = val
    autoclave_gwp_min = autoclave_gwp/min_auto
    autoclave_gwp_max = autoclave_gwp/max_auto
    min_max_lst = [min - autoclave_gwp_min, max - autoclave_gwp_max]
    min_max_auto = [min_auto, max_auto]

    return min_max_lst, min_max_auto

def case2_initilazation(df_be, database_type, autoclave_gwp):

    idx_sens = [
            'autoclave',
            'cabinet washer',
            'Life time',
            'sterilization',
            'surgery time',
            'total'
        ]



    val_dct = {
        'autoclave' : {},
        'surgery time' : {},
        'sterilization' : {},
        'Life time' : {},
        'cabinet washer' : {}
    }
    col_to_df = []

    min_max_lst, min_max_auto = sterilization_min_max(database_type, autoclave_gwp)

    use_elec_var = [((60-2)/60*40 + 500 * 2/60)/1000, ((60-10)/60*40 + 500 * 10/60)/1000]
    for idx in df_be.index:
        if 'SUD' in idx:
            val_dct['surgery time'].update({idx : [use_elec_var[0], use_elec_var[1]]})
        else:
            val_dct['Life time'].update({idx : [50, 500]})
            val_dct['autoclave'].update({idx : min_max_auto})
            val_dct['cabinet washer'].update({idx : [32, 48]})
            val_dct['sterilization'].update({idx : min_max_lst})
            val_dct['surgery time'].update({idx : [use_elec_var[0], use_elec_var[1]]})


        col_to_df.append(f'{idx} - lower%')
        col_to_df.append(f'{idx} - upper%')

    df = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    return df, val_dct, idx_sens, col_to_df

def row_total_calculator(df_sensitivity):
    tot_lst = [0] * len(df_sensitivity.columns)
    for tot, col in enumerate(df_sensitivity.columns):
        for idx, row in df_sensitivity.iterrows():
            if idx != 'total':
                tot_lst[tot] += row[col]
                row[col] = f"{row[col]:.2f}%"
                if row[col] == "-0.00%" or row[col] == "0.00%":
                    row[col] = "-"

    for tot, col in enumerate(df_sensitivity.columns):
        for idx, row in df_sensitivity.iterrows():
            if idx == 'total':
                row[col] = f"{tot_lst[tot]:.2f}%"
    return df_sensitivity

def uncertainty_values_new(variables, path, autoclave_gwp):
    database_name, df_GWP, db_type, save_dir, impact_category, flows = variables
    if 'case1' in database_name:
            flow_legend = [
                        'H2S',
                        'H2R',
                        'ASC',
                        'ASW',
                        'H4S',
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
        df, val_dct, idx_sens, col_to_df = case1_initilazation(df_be)
        return uncertainty_case1(df, val_dct, df_be, totals_df, idx_sens, col_to_df)
    
    elif 'case2' in database_name:
        
        df, val_dct, idx_sens, col_to_df = case2_initilazation(df_be, db_type, autoclave_gwp)
        return uncertainty_case2(val_dct, df_be, totals_df, idx_sens, col_to_df)



def save_sensitivity_to_excel(variables, path, autoclave_gwp_dct):
    identifier = variables[0]
    save_dir = variables[3]
    df_sens = uncertainty_values_new(variables, path, autoclave_gwp_dct)

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


def obtain_case1_autoclave_gwp(variables, path):
    autoclave_gwp_dct = {}
    for key, item in variables.items():
        if '1' in key:
            autoclave_gwp_dct[f'case2_{item[2]}'] = autoclave_gwp_impact(item, path)
            autoclave_gwp_dct[item[0]] = ''
            
    return autoclave_gwp_dct

def iterative_save_sensitivity_results_to_excel(variables, path):
    autoclave_gwp_dct = obtain_case1_autoclave_gwp(variables, path)

    for key, item in variables.items():
        if '1' in key:
            save_sensitivity_to_excel(item, path, autoclave_gwp_dct[key])
        elif '2' in key:
            save_sensitivity_to_excel(item, path, autoclave_gwp_dct[key])



