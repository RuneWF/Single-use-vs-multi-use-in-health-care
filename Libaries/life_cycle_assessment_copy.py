import pandas as pd
import json
import copy
import re

# Import BW25 packages
import bw2data as bd
import brightway2 as bw 
import bw2calc as bc

# Importing self-made libaries
from standards import *
import LCA_plots as lp
import non_bio_co2 as nbc
import import_ecoinvent_and_databases as ied

def get_all_flows(path, lcia_meth, bw_project="single use vs multi use"):
    bd.projects.set_current("single use vs multi use")
    db_type = ['apos', 'consq', 'cut_off']

    flows = {}
    save_dir = {}
    database_name_dct = {}
    file_name = {}
    db_type_dct = {}
    flow_legend = {}
    file_name_unique_process = {}
    sheet_name = {}
    initialization = {}
    for nr in range(1,3):


        # Setting brightway to the given project

        # Specifiyng which database to usey

        for tp in db_type:
            database_name = f'case{nr}' + '_' + tp
            database_name_dct[database_name] = database_name
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
                flow_leg = [
                    'H2S',
                    'H2R',
                    'ASC',
                    'ASW',
                    'H4S',
                    'H4R',
                    'ALC',
                    'ALW'
                    ]
                sheet_name[database_name] = 'case1'
            elif 'case2' in str(db):
                for act in db:
                    temp = act['name']
                    if temp == 'SUD' or temp == 'MUD':
                        flow.append(temp)
                    flow_leg = ['SUD', 'MUD']
                    sheet_name[database_name] = 'case2'
                flow.sort()
                flow.reverse()
            flows[database_name] = flow
            dir_temp = results_folder(path+'\\results', f"case{nr}", tp)
            save_dir[database_name] = dir_temp
            file_name[database_name] = f'{dir_temp}\data_case{nr}_{tp}_recipe.xlsx'
            db_type_dct[database_name] = tp
            flow_legend[database_name] = flow_leg
            # print(f'{dir_temp}\data_uniquie_case{nr}_{tp}_{lcia_meth}.xlsx')
            file_name_unique_process[database_name] = f'{dir_temp}\data_uniquie_case{nr}_{tp}_{lcia_meth}.xlsx'
            initialization[database_name] = [bw_project, database_name, flow, lcia_meth, tp]
    lst = [save_dir, file_name, flow_legend, file_name_unique_process, sheet_name]
    
    return lst, initialization

def initilization(path, lcia_method, ecoinevnt_paths, system_path, bw_project="single use vs multi use"):
    ied.database_setup(ecoinevnt_paths, system_path)
    ui = int(input(f'Select 1 to choose flows based on the LCA type, 2 for choosing them yourself and 3 for everything'))
    if ui == 1 or ui == 2:
        ui1 = int(input('select 1 for case1 and 2 for case2'))

        # Specifying if it is CONSQ (consequential) or APOS
        ui2 = int(input('select 1 for apos and 2 for consequential and 3 for cut off'))
        if ui2 == 1:
            db_type = 'apos'
        elif ui2 == 2:
            db_type = 'consq'
        elif ui2 == 3:
            db_type = 'cut_off'
        
        database_name = f'case{ui1}' + '_' + db_type

        # Creating the flow legend
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
            file_identifier = 'case1'
            
        else:
            flow_legend = ['SUD', 'MUD']
            file_identifier = 'case2'

        # Specifying the file name and sheet name
        
        sheet_name = f'{file_identifier}'

        # Creating the saving directory for the results
        save_dir = results_folder(path+'\\results', file_identifier, db_type)
        file_name = f'{save_dir}\data_{file_identifier}_{db_type}_{lcia_method}.xlsx'
        ui2 = int(input(f'Select 1 to choose flows based on {db_type}, else 2 for choosing them yourself'))
        if ui2 == 1:
            flows = get_database_type_flows(database_name)
        elif ui2 == 2:
            flows = get_user_specific_flows(database_name)
        
        print('Chosen flows:')
        for f in flows:
            print(f)

        initialization = [bw_project, database_name, flows, lcia_method, db_type]
        file_name_unique_process = f'{save_dir}\data_uniquie_{file_identifier}_{db_type}_{lcia_method}.xlsx'

    else:
        lst, initialization = get_all_flows(path, lcia_method)
        save_dir, file_name, flow_legend, file_name_unique_process, sheet_name = lst

    return flow_legend, file_name, sheet_name, save_dir, initialization, file_name_unique_process

# Function to obtain the LCIA category to calculate the LCIA results
def lcia_method(method):
    # Checking if the LCIA method is ReCiPe, and ignores difference between lower and upper case 
    if 'recipe' in method.lower():
        # Using H (hierachly) due to it has a 100 year span
        # Obtaining the midpoint categpries and ignoring land transformation (Land use still included)
        nbc.remove_bio_co2_recipe()
        all_methods = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H) Runes edition' in str(m) and 'no LT' not in str(m)] # Midpoint

        # Obtaining the endpoint categories and ignoring land transformation
        endpoint = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H) Runes edition' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

        # Combining midpoint and endpoint, where endpoint is added to the list of the midpoint categories
        for meth in endpoint:
            all_methods.append(meth)
            
        print('Recipe is selected')

    # Checking if EF is choses for the LCIA method
    elif 'ef' in method.lower():
        all_methods = [m for m in bw.methods if 'EF v3.1 EN15804' in str(m) and "climate change:" not in str(m)]
        print('EF is selected')

    else:
        print('Select either EF or ReCiPe as the LCIA methos')
        all_methods = []

    # Returning the selected LCIA methods
    return all_methods

# Function to extract the flows to calculate the LCIA for
def get_database_type_flows(database: str):
    # Setting brightway to the given project
    db = bd.Database(database)
    # Specifiyng which database to usey
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
            
    # Returning the flows
    return flow

def get_user_specific_flows(database: str):
    # Setting brightway to the given project

    # Specifiyng which database to use
    db = bd.Database(database)

    chosen_flows = []
    # Letting the user specify which flows shall be calculated
    print("choose 'y' if you want to calculate for this flow, and 'n' if not")
    for act in db:
        flow = act['name']
        user_input = input(f'Do you want to calculate for {flow}? [y/n]')
        if 'y' in user_input.lower():
            chosen_flows.append(flow)
        elif 'n' in user_input.lower():
            pass
        else:
            print('Choose either y or n')

    chosen_flows.sort()

    return chosen_flows

# Function to initialize parameters for the LCIA calculations
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
    impact_category = lcia_method(method)
    
    # obtaing a shortened version of the impact categories for the plots
    plot_x_axis = [0] * len(impact_category)
    for i in range(len(plot_x_axis)):
        plot_x_axis[i] = impact_category[i][2]

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
    print('Initialization is completed')
    return FU, impact_category

# saving the LCIA results to excel
def save_LCIA_results(df, file_name, sheet_name, impact_category):
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

# Function to import the LCIA results from excel
def import_LCIA_results(file_name, flow, impact_category):
    
    if type(impact_category) == tuple:
        impact_category = [impact_category]
    
    # Reading from Excel
    df = pd.read_excel(file_name)

    # Convert JSON strings back to lists for all columns
    df = df.map(lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x)
    # Setting the index to the flow
    df = df.set_axis(flow, axis=0)

    # Updating column names
    df.columns = impact_category

    # Return the imported dataframe
    return df

# Function to seperate the midpoint and endpoint results for ReCiPe
def recipe_dataframe_split(df):
    # Obtaining the coluns from the dataframe
    col_df = df.columns
    col_df = col_df.to_list()

    # Seperating the dataframe into one for midpoint and another for endpoint
    df_midpoint = df[col_df[:-3]]
    df_endpoint = df[col_df[-3:]]
    

    return df_midpoint, df_endpoint

# Function to create two dataframes, one where each subprocess' in the process are summed 
# and the second is scaling the totals in each column to the max value
def dataframe_element_scaling(df):
    # Creating a deep copy of the dataframe to avoid changing the original dataframe
    df_tot = copy.deepcopy(df)

    # Obating the sum of each process for each given LCIA category
    for col in range(df.shape[1]):  # Iterate over columns
        for row in range(df.shape[0]):  # Iterate over rows
            tot = 0
            for i in range(len(df.iloc[row,col])):
                tot += df.iloc[row,col][i][1]
            df_tot.iloc[row,col] = tot

    df_cols = df_tot.columns
    df_cols = df_cols.to_list()

    #df_norm = pd.DataFrame().reindex_like(df_tot) #https://stackoverflow.com/questions/23195250/create-empty-dataframe-with-same-dimensions-as-another
    df_scaled = copy.deepcopy(df_tot)

    # Obtaing the scaled value of each LCIA results in each column to the max
    for i in df_cols:
        scaling_factor = max(abs(df_scaled[i]))
        # print(df_tot[i])
        for idx, row in df_scaled.iterrows():
            row[i] /= scaling_factor

    return df_tot, df_scaled

# Normiliation for EF
def LCIA_normalization(directory, df):
    file = f'{directory}Single-use-vs-multi-use-in-health-care\\Norm + Weigh.xlsx'
    data_NW = pd.read_excel(file)
    columns = df.columns

    norm_lst = data_NW['Normalization'].tolist()
    weigh_lst = data_NW['Weighting'].tolist()

    weigh_df = pd.DataFrame().reindex_like(df)

    counter = 0

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

# Obtaining the uniquie elements to determine the amount of colors needed for the plots
def unique_elements_list(database_name):
    category_mapping = lp.category_organization(database_name)
    unique_elements = []
    for item in category_mapping.values():
        for ilst in item:
            unique_elements.append(ilst)

    return unique_elements

def rearrange_dataframe_index(df, database):
    idx_dct = {}
    idx_lst = df.index
    if 'case1' in database:
        plc_lst = [1, 0, 5, 4, 6, 7, 2, 3]

        # Letting the user decide the new order of the index
        for plc, idx in enumerate(df.index):
            idx_dct[idx] = plc_lst[plc]
            
        # Creating the new index list
        idx_lst = [''] * len(idx_dct)
        for key, item in idx_dct.items():
            idx_lst[item] = key

        impact_category = df.columns
        df_rearranged = pd.DataFrame(0, index=idx_lst, columns=impact_category, dtype=object)


        # Arranging the dataframe to the new dataframe
        for icol, col in enumerate(impact_category):
            for row_counter, idx in enumerate(df_rearranged.index):
                rearranged_val = df.at[idx, col] # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html#pandas.DataFrame.at
                df_rearranged.iloc[row_counter, icol] = rearranged_val
        return df_rearranged
    else:
        # If rearrnge is not chosen it returns the same dataframe as inputtet
        return df

# Function to rearrange the data in the dataframe
def rearrange_dataframe_index_user_specific(df):
    rearrange = True if input('Do you want to rearrange the data? [y/n]') == 'y' else False
    # Checking if rearrange is chosen
    if rearrange == True:
        idx_dct = {}
        idx_lst = df.index

        # Specifying the amount of placement for the index
        plc_lst = [new_plc for new_plc in range(len(idx_lst))]

        # Letting the user decide the new order of the index
        for idx in df.index:
            user = int(input(f'What placement shall {idx} have in the graph [{plc_lst}]'))
            idx_dct[idx] = user
            for plc in plc_lst:
                # removing the chosen placement
                if user == plc:
                    plc_lst.remove(plc)

        # Creating the new index list
        idx_lst = [''] * len(idx_dct)
        for key, item in idx_dct.items():
            idx_lst[item] = key

        impact_category = df.columns
        df_rearranged = pd.DataFrame(0, index=idx_lst, columns=impact_category, dtype=object)


        # Arranging the dataframe to the new dataframe
        for icol, col in enumerate(impact_category):
            for row_counter, idx in enumerate(df_rearranged.index):
                rearranged_val = df.at[idx, col] # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html#pandas.DataFrame.at
                df_rearranged.iloc[row_counter, icol] = rearranged_val
                # row_counter += 1
                
                # if row_counter == len(idx_dct):  # Reset when all flows have been processed in the column
                #     row_counter = 0
        return df_rearranged
    else:
        # If rearrnge is not chosen it returns the same dataframe as inputtet
        return df

def dataframe_results_handling(df, database_name, plot_x_axis_all, lcia_meth):
    df_rearranged = rearrange_dataframe_index(df, database_name)
    if 'recipe' in lcia_meth:
        df_res, df_endpoint = recipe_dataframe_split(df_rearranged)
        plot_x_axis_end = plot_x_axis_all[-3:]
        ic_mid = plot_x_axis_all[:-3] 
        plot_x_axis = []
        for ic in ic_mid:
            string = re.findall(r'\((.*?)\)', ic)
            if 'ODPinfinite' in  string[0]:
                string[0] = 'ODP'
            elif '1000' in string[0]:
                string[0] = 'GWP'
            plot_x_axis.append(string[0])

        return [df_res, df_endpoint], [plot_x_axis, plot_x_axis_end]
    else:
        df_res = df_rearranged
        plot_x_axis = plot_x_axis_all

        return df_res, plot_x_axis
