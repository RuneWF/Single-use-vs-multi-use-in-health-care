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

from standards import *

def lcia_method(method):
    if 'recipe' in method.lower():
        all_methods = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H)' in str(m) and 'no LT' not in str(m)]
        endpoint = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H)' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

        for meth in endpoint:
            all_methods.append(meth)

        print('Recipe is selected')

    elif 'ef' in method.lower():
        all_methods = [m for m in bw.methods if 'EF v3.1 EN15804' in str(m) and "climate change:" not in str(m)]
        print('EF is selected')
    else:
        print('Select either EF or ReCiPe as the LCIA methos')
        all_methods = []

    return all_methods

def LCA_initialization(name: str, db: str, flows: list, method: str, db_type: str) -> tuple:

    if 'CONSQ' in db_type and 'Ofir' in name:
        db_eco = 'Consq EcoInvent'
    elif 'APOS' in db_type and 'Ofir' in name:
        db_eco = 'APOS EcoInevnt'
    else:
        db_eco = 'Consequential'

    print(db_eco)

    db_cyl = 'Cylinder'
    db_pellet = 'Pellet'
    bd.projects.set_current(name)

    bi.bw2setup()
    eidb = bd.Database(db)
    eidb_consq = bd.Database(db_eco)
    eidb_cyl = bd.Database(db_cyl)
    eidb_pellet = bd.Database(db_pellet)

    procces_keys = {key: None for key in flows}

    size = len(flows)
    all_acts = list(eidb) + list(eidb_consq) + list(eidb_cyl) + list(eidb_pellet)

    for act in all_acts:
        for proc in range(size):
            if act['name'] == flows[proc]:
                procces_keys[flows[proc]] = act['code']

    process = []
    key_counter = 0

    for key, item in procces_keys.items():
        try:
            if eidb.get(item) in eidb:
                process.append(eidb.get(item))
                # print(f'Process "{key}" found in main database (eidb): {eidb.get(item)}')
            else:
                copied_process = copy_process(item, eidb_consq, eidb)
                if copied_process:
                    process.append(copied_process)
                    print('Process copied')
                else:
                    print(f"Process with key '{item}' not found in the consequential database (eidb_consq) either.")
        except KeyError:
            print(f"Process with key '{item}' not found in the database '{eidb}'")
            process = None
        key_counter += 1

    products_list = []
    
    if process:
        for proc in process:
            for exc in proc.exchanges():
                if exc['type'] == 'production':
                    products_list.append(exc.input)

    linked_processes_list = []

    if process:
        for proc in process:
            for exc in proc.exchanges():
                linked_processes_list.append(exc.input)

            linked_processes_list = list(set(linked_processes_list))

        proc_keys = {}
        name_keys = {}

        for linked_process in linked_processes_list:
            if linked_process[0] not in proc_keys:
                proc_keys[linked_process[0]] = []
                name_keys[linked_process[0]] = []
            proc_keys[linked_process[0]].append(linked_process[1])
            name_keys[linked_process[0]].append(linked_process)

    # all_methods = [m for m in bw.methods if 'EF v3.1 EN15804' in str(m)]
    # filtered_methods = [method for method in all_methods if "climate change:" not in method[1]]

    impact_category = lcia_method(method)
    
    plot_x_axis = [0] * len(impact_category)
    for i in range(len(plot_x_axis)):
        plot_x_axis[i] = impact_category[i][2]

    product_details = {}
    product_details_code = {}

    if process:
        for proc in process:
            product_details[proc['name']] = []
            product_details_code[proc['name']] = []

            for exc in proc.exchanges():
                if 'Use' in exc.output['name'] and exc['type'] == 'biosphere':
                    product_details[proc['name']].append({exc.input['name']: [exc['amount'], exc.input]})
                    if exc.input in eidb or exc.input in eidb_consq or exc.input in eidb_cyl:
                        product_details_code[proc['name']].append([exc.output, exc.output['name'], exc.output['code'], exc['amount']])
                elif exc['type'] == 'technosphere':
                    product_details[proc['name']].append({exc.input['name']: [exc['amount'], exc.input]})
                    if exc.input in eidb or exc.input in eidb_consq or exc.input in eidb_cyl:
                        product_details_code[proc['name']].append([exc.input, exc.input['name'], exc.input['code'], exc['amount']])

    idx_df = []
    fu_val = []
    p_code = []

    for process_name, details in product_details.items():
        for detail in range(len(details)):
            for key, item in details[detail].items():
                idx_df.append(key)
                fu_val.append(details[detail][key][0])
                p_code.append(details[detail][key])

    FU_proc = []

    for flow in flows:
        for flow_length in range(len(product_details[flow])):
            for key in product_details[flow][flow_length].keys():
                if flow in key:
                    key = key.replace(f'{flow} ', '')
                FU_proc.append(key)

    FU = []
    for key, item in product_details.items():
        for idx in item:
            for n, m in idx.items():
                FU.append({key: {m[1]: m[0]}})

    print('Initialization is completed')
    return FU, FU_proc, impact_category, plot_x_axis, product_details_code

def copy_process(process_code: str, eidb_consq, eidb):
    try:
        external_process = eidb_consq.get(process_code)
        if external_process:
            new_process = eidb.new_activity(
                code=external_process['code'],
                name=external_process['name'],
                unit=external_process['unit'],
                location=external_process['location'],
            )
            for exc in external_process.exchanges():
                new_process.new_exchange(
                    input=exc.input,
                    output=new_process,
                    type=exc['type'],
                    amount=exc['amount'],
                    unit=exc['unit']
                )
            new_process.save()
            print(f"Process '{external_process['name']}' copied from eidb_consq to eidb.")
            return new_process
    except Exception as e:
        print(f"Error copying process from eidb_consq: {e}")
    return None

def life_cycle_impact_assessment(flows, functional_unit, impact_categories, process):
    print()
    print('Calculating the LCA results:')
    if type(impact_categories) == tuple:
        impact_categories = [impact_categories]

    # Define the dimensions
    n = len(flows)  # number of rows (flows)
    m = len(impact_categories)  # number of columns (impact categories)

    # Create a DataFrame to store results
    df = pd.DataFrame(0, index=flows, columns=impact_categories, dtype=object)  # dtype=object to handle lists

    calc_count = 1
    row_counter = 0

    # Loop through impact categories
    for col, impact in enumerate(impact_categories):
        # Loop through flows
        for f in flows:
            df_lst = []  # Clear list for each flow in each impact category
            for func_unit in range(len(functional_unit)):
                for FU_key, FU_item in functional_unit[func_unit].items():
                    # cat = impact_categories[col]
                    if f in FU_key:
                        # Perform LCA
                        lca = bw.LCA(FU_item, impact)
                        lca.lci()
                        lca.lcia()
                        if len(process) == 1:
                            df_lst.append([process, lca.score])
                        else:
                            df_lst.append([process[func_unit], lca.score])

                        # Print progress
                        print(f"Calculation {calc_count} of {m * len(functional_unit)}", FU_item, impact[1], lca.score)
                        calc_count += 1
            
            # Assign the list of results to the DataFrame
            df.iloc[row_counter, col] = df_lst
            # Update the row counter
            row_counter += 1
            
            print(f'{impact[1]} at row {row_counter - 1} col {col} has been assigned the list {df_lst}')

            
            if row_counter == len(flows):  # Reset when all flows have been processed
                row_counter = 0
        
    return df

def save_LCA_results(df, file_name, sheet_name, impact_category):
    if type(impact_category) == tuple:
        impact_category = [impact_category]
    # Convert each cell to a JSON string for all columns
    df_save = df.map(lambda x: json.dumps(x) if isinstance(x, list) else x)

    # Save to Excel
    with pd.ExcelWriter(file_name) as writer:
        df_save.to_excel(writer, sheet_name=sheet_name, index=False, header=True)

    print('DataFrame with nested lists written to Excel successfully.')

    with open("impact_categories", "w") as fp:
        json.dump(impact_category, fp)

def import_LCA_results(file_name, flow, impact_category):
    # Reading from Excel
    if type(impact_category) == tuple:
        impact_category = [impact_category]

    df = pd.read_excel(file_name)

    # Convert JSON strings back to lists for all columns
    df = df.map(lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x)
    df = df.set_axis(flow)
    df.columns = impact_category

    return df

def recipe_dataframe_split(df):
    col_df = df.columns
    col_df = col_df.to_list()

    df_midpoint = df[col_df[:-3]]

    df_endpoint = df[col_df[-3:]]
    

    return df_midpoint, df_endpoint

def process_filter(FU):
    filtered_dict = {}
    for i, scenario in enumerate(FU):
        for sc_key, sc_item in scenario.items():

            for sc_proc_key, sc_proc_item in sc_item.items():
                
                # N2O
                if 'Consequential' in sc_proc_key[0] or 'Consq EcoInvent' in sc_proc_key[0] or 'APOS EcoInevnt' in sc_proc_key[0]:
                    filtered_dict[sc_proc_key] = sc_proc_item 
                    # print(sc_item)
                # Avoided energy and heat
                elif 'no Energy Recovery' in f'{sc_proc_key}' or 'heating grid' in f'{sc_proc_key}':
                    # print(f'Key : {sc_proc_key}, Item : {sc_proc_item}, {sc_key}')
                    filtered_dict[sc_proc_key] = sc_proc_item 
                    # print(sc_item)
    return filtered_dict

def obtaining_sub_process(sub_product_details):
    sub_proccess = {}
    amount = {}
    for key, details in sub_product_details.items():
        # print(f"Process: {key}")

        sub_proccess[key] = []

        for detail in details:
            
            sub_proccess[key].append([detail[0], detail[1], detail[3]])
            amount[detail[1]] = []
            amount[detail[1]].append(detail[3])
    return sub_proccess, amount

def sub_process_initilization(sub_proccess, FU, name, idx_name, method, db_type):

    filtered_dict = process_filter(FU)
    # Initializing empty dictionaries to store the results
    FU_sub = {key: [] for key in sub_proccess}
    FU_sub_proc = {key: [] for key in sub_proccess}


    for proc, sub_proc in sub_proccess.items():
        # print(f'Process: {proc}')
        temp = {}
        fu_temp = []
        for proc_idx in range(len(sub_proc)):
            #print(sub_proc[proc_idx])
            flow = [sub_proc[proc_idx][1]]
            
            db_proc = sub_proc[proc_idx][0][0]
            # print(f'Flow : {flow}, Database: {db_proc}, Subprocess : {sub_proc}')
            if db_proc == 'Consequential' or 'APOS' in db_proc  or db_proc == 'Consq EcoInvent' and sub_proc[proc_idx][0] in filtered_dict:
                #print(flow)
                fu = [{flow[0] : filtered_dict}]
                p = flow
            # elif 'no Energy Recovery' in f'{flow}' or 'heating grid' in f'{flow}':
            #     fu = [{flow[0] : filtered_dict}]
            #     p = flow
            else:
                fu, p, ic, pxa, kokos = LCA_initialization(name, db_proc, flow, method, db_type)

            
            temp[flow[0]] = []
            temp[flow[0]].append(p)
            for fuck in fu:
                fu_temp.append(fuck)

        FU_sub[proc].append(fu_temp)
        FU_sub_proc[proc].append(temp)

    idx = []
    sc_counter = 1
    for k, i in FU_sub_proc.items():
        for kk, ii in i[0].items():
            idx.append(kk + f' - sc {sc_counter}')
        sc_counter += 1

    with open(idx_name, "w") as fp:
        json.dump(idx, fp)

    return FU_sub, FU_sub_proc, idx

def FU_contibution_initilization(FU_sub, FU_sub_proc):
    flow_count = 0
    flow_sub = []
    functional_unit_sub = []
    for key, item in FU_sub_proc.items():
        # print(key)
        df_temp = {}
        for pommesfrit in item:
            for pom_process, pom_subprocess in pommesfrit.items():
                for pompom in pom_subprocess:

                    fu_proc_temp = pom_process
                    fu_sub_proc_temp = pompom
                    fu_temp = FU_sub[key][0]

                    flow_sub.append(fu_proc_temp)
            functional_unit_sub.append(fu_temp)

    for func_unit in functional_unit_sub:
        flow_count += len(func_unit) 

    return flow_count, flow_sub, functional_unit_sub     

def process_update(FU, FU_sub):
    functional_unit_sub_new = copy.deepcopy(FU_sub)
    updated_keys = set()

    for fcu in range(len(FU_sub)):
        for fu_ind in range(len(FU_sub[fcu])):
            for fu_ind_key, fu_ind_item in FU_sub[fcu][fu_ind].items():
                funky_key = [i for i in fu_ind_item.keys()][0]
                for fu_sc in range(len(FU)):
                    for uuuu, fu_sc_val in FU[fu_sc].items():
                        funky_key_sc = [i for i in fu_sc_val.keys()][0]
                    # if fu_ind_key not in updated_keys:
                    #     print(fu_ind_key)
                        if fu_ind_key in f'{funky_key_sc}' and 'biosphere3' in funky_key[0]:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'PE incineration no Energy Recovery' in f'{funky_key_sc}' and 'PE incineration no Energy Recovery' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'PP incineration no Energy Recovery' in f'{funky_key_sc}' and 'PP incineration no Energy Recovery' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'heating grid' in f'{funky_key_sc}' and 'heating grid' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'transport Alu' in f'{funky_key_sc}' and 'transport Alu' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'transport Pla' in f'{funky_key_sc}' and 'transport Pla' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'autoclave ' in f'{funky_key_sc}' and 'autoclave ' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)                            
                        elif 'disinfection' in f'{funky_key_sc}' and 'disinfection' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'Handwash' in f'{funky_key_sc}' and 'Handwash' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'energy avoided' in f'{funky_key_sc}' and 'energy avoided' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)

    fu_sub_updated = []
    for scenario in range(len(functional_unit_sub_new)):
        temp_lst = []  # Move temp_lst initialization here
        for proc in range(len(functional_unit_sub_new[scenario])):
            temp = functional_unit_sub_new[scenario][proc]
            
            if temp not in temp_lst:
                temp_lst.append(temp)
        
        fu_sub_updated.append(temp_lst)  # Append temp_lst after the inner loop

    return fu_sub_updated


def LCIA_contribution(impact_category, flow_count, sub_proc, FU_sub, amount, idx):
    if type(impact_category) == tuple:
        impact_category = [impact_category]

    df_cont = pd.DataFrame(0, index=idx, columns=impact_category, dtype=object)  # dtype=object to handle lists

    calc = len(impact_category)*flow_count
    dct = {}
    row_counter = 0
    calc_count = 1
    
    # Iterate over impact categories (columns)
    for column, cat in enumerate(impact_category):
        # Iterate over processes and their corresponding flows in FU_sub_proc
        for k, i in sub_proc.items():
            # For each flow in the current process
            
            for f in i[0].keys():
                accounted_flows = []
                
                print(f"Processing flow: {f} in impact category: {cat[1]}")

                # Initialize the result list for the current flow
                dct[f] = []
                df_lst = []

                # Perform LCA for each functional unit
                for func_unit in range(len(FU_sub)):
                    
                    for FU_dict in FU_sub[func_unit]:
                        for  dk, di in FU_dict.items():
                            # print(dk, di)
                            if dk in f and di.keys() not in accounted_flows:
                                
                                accounted_flows.append(di.keys())
                                FU_dict_copy = copy.deepcopy(FU_dict)

                                # Update the flow amounts
                                for key, item in FU_dict.items():
                                    for FU_key, FU_val in item.items():
                                        FU_dict_copy[key][FU_key] = FU_dict[key][FU_key] * amount[f][0]
                    
                                # Perform LCA
                                lca = bw.LCA(FU_dict_copy[key], cat)
                                lca.lci()
                                lca.lcia()

                                # Append the result (using the temp variable for functional unit sub-process)
                                df_lst.append([f'{FU_key}', lca.score])
                                print(f"{FU_key} Calculation {calc_count} of {calc}, Score: {lca.score} {cat[1]}")
                                calc_count += 1

                
                

                # # Assign the result list to the DataFrame for the current flow and column (impact category)
                df_cont.iloc[row_counter, column] = df_lst
                # Update the row counter after processing all flows in the current impact category
                row_counter += 1
                print(f'row : {row_counter - 1}, col : {column} is assigned list : {df_lst}')




                # Reset the row counter if it reaches the number of rows (flows)
                if row_counter == len(idx):
                    row_counter = 0

    return df_cont

def dataframe_element_scaling(df_test):
    df_tot = copy.deepcopy(df_test)

    for col in range(df_test.shape[1]):  # Iterate over columns
        for row in range(df_test.shape[0]):  # Iterate over rows
            tot = 0
            for i in range(len(df_test.iloc[row,col])):
                #print(df_updated.iloc[row,col][i][1])
                tot += df_test.iloc[row,col][i][1]
            df_tot.iloc[row,col] = tot
            # print('New row')

    df_cols = df_tot.columns
    df_cols = df_cols.to_list()

    #df_norm = pd.DataFrame().reindex_like(df_tot) #https://stackoverflow.com/questions/23195250/create-empty-dataframe-with-same-dimensions-as-another
    df_norm = copy.deepcopy(df_tot)


    for i in df_cols:
        scaling_factor = max(df_norm[i])
        # print(df_tot[i])
        for idx, row in df_norm.iterrows():
            row[i] /= scaling_factor
        # df_norm = copy.deepcopy(df_tot)

    return df_tot, df_norm

def dataframe_column_structure(impact_category):
    if type(impact_category) == tuple:
        impact_category = [impact_category]

    plot_x_axis = [0] * len(impact_category)
    for i in range(len(plot_x_axis)):
        if "photochemical oxidant formation" in impact_category[i][1]:
            plot_x_axis[i] = "Photochemical Oxidant Formation"
        else:
            plot_x_axis[i] = impact_category[i][1].title()

    return plot_x_axis

def LCIA_normalization(directory, df):
    file = f'{directory}Single-use-vs-multi-use-in-health-care\\Norm + Weigh.xlsx'
    data_NW = pd.read_excel(file)
    columns = df.columns

    norm_lst = data_NW['Normalization'].tolist()
    weigh_lst = data_NW['Weighting'].tolist()

    weigh_df = pd.DataFrame().reindex_like(df)

    counter = 0
    weigh_df_copy = copy.deepcopy(weigh_df)
    for i in columns:
        for j, row in df.iterrows():
            nw_val = row[i] * norm_lst[counter] * weigh_lst[counter]
            weigh_df.at[j, i] = nw_val #https://saturncloud.io/blog/how-to-update-a-cell-value-in-pandas-dataframe/
            # print(j, nw_val, row[i] * norm_lst[counter] * weigh_lst[counter])
        counter += 1
        
    # print(weigh_df)
    lst = []
    for idx_val, row in weigh_df.iterrows():
        temp = 0
        for i in columns:
        
        
            temp += row[i]
        lst.append(temp)

    lst_norm_weighted = [0] * len(lst)
    lst_max = max(lst)

    for n in range(len(lst)):
        if lst_max != 0:
            lst_norm_weighted[n] = lst[n] / lst_max

    return lst_norm_weighted

def unique_elements_list(df_GWP, db_type):
    unique_elements = []
    for col in df_GWP.columns:
        for i, row in df_GWP.iterrows():
            for elm in row[col]:
                x = elm[0]
                gwp = elm[1]
                if f'- {db_type}' in x:
                    #print(key)
                    x = x.replace(f' - {db_type}', '')
                if 'alubox' in x:           
                    x = x.replace('alubox ', '')
                    if 'raw' in x and 'avoid' not in x.lower():
                        x = x.replace('raw materials', 'Raw mat.')
                    elif 'raw' in x and 'avoid' in x:
                        x = 'Avoided virgin mat.'
                    if 'production' in x:
                        x = 'Production'
                if 'Waste' in x:
                    x = 'Incineration'
                if 'energy avoided' in x:
                    x = 'Avoided electricity'
                if 'heating' in x:
                    x = 'Avoided heat'
                if 'market for polypropylene' in x:
                    if gwp < 0:
                        x = 'Avoided PP'
                    else:
                        x = 'PP granulate'
                if 'PE granulate' in x:
                    if gwp < 0:
                        x = 'Avoided PE'
                    else:
                        x = 'PE granulate'
                if 'no Energy Recovery' in x:
                    x = x.replace(' no Energy Recovery', '')

                if 'board box' in x:
                    x = 'Cardboard box'
                if 'packaging film' in x:
                    x = 'PE packaging film prod.'
                if 'pp' in x:
                    x = x.replace('pp', 'PP')
                if 'autoclave' in x:
                    x = x.replace('autoclave', 'Autoclave')
                if 'transport' in x:
                    x = 'Transport'

                if x not in unique_elements:
                    unique_elements.append(x)
                    # print(elm[0])
    unique_elements.append('Total')

    return unique_elements

def round_down(value):
    # https://stackoverflow.com/questions/41383787/round-down-to-2-decimal-in-python
    value = math.floor(value * 10)/10.0
    return value