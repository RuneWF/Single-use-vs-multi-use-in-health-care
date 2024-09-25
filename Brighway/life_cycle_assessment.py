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

def LCA_initialization(name: str, db: str, flows: list) -> tuple:
    db_consq = 'Consequential'
    db_cyl = 'Cylinder'
    db_pellet = 'Pellet'
    bd.projects.set_current(name)

    bi.bw2setup()
    eidb = bd.Database(db)
    eidb_consq = bd.Database(db_consq)
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

    all_methods = [m for m in bw.methods if 'EF v3.1 EN15804' in str(m)]
    filtered_methods = [method for method in all_methods if "climate change:" not in method[1]]
    removed_methods = [method[1] for method in all_methods if "climate change:" in method[1]]

    impact_category = filtered_methods
    
    plot_x_axis = [0] * len(impact_category)
    for i in range(len(plot_x_axis)):
        plot_x_axis[i] = impact_category[i][1]

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
