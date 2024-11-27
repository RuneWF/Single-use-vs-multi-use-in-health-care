import pandas as pd
import json
import copy

# Import BW25 packages
import bw2data as bd
import bw2io as bi
import brightway2 as bw 
import bw2calc as bc

# Importing self-made libaries
from standards import *
import LCA_plots as lp
import non_bio_co2 as nbc

def initilization(path, lcia_method):
    # Specifying if it is CONSQ (consequential) or APOS
    ui1 = int(input('select 0 for APOS and 1 for CONSQ'))
    if ui1 == 0:
        db_type = 'APOS'
    elif ui1 == 1:
        db_type = 'CONSQ'
    
    # Let the use decide the project and database
    database_project, database_name = select_project_and_database()

    # Creating the flow legend
    if 'sterilization' in database_name:
        flow_legend = ['H2S',
                'H2R',
                'ASC',
                'ASW',
                'H4S',
                'H4R',
                'ALC',
                'ALW'
                ]
        file_identifier = 'sterilization'
        
    else:
        flow_legend = ['SUD', 'MUD']
        file_identifier = 'diathermy'

    # Specifying the file name and sheet name
    
    sheet_name = f'{file_identifier}'

    # Creating the saving directory for the results
    save_dir = results_folder(path+'\\results', file_identifier, db_type)
    file_name = f'{save_dir}\data_{file_identifier}_{db_type}_{lcia_method}.xlsx'
    ui2 = int(input(f'Select 0 to choose flows based on {db_type}, else 1 for choosing them yourself'))
    if ui2 == 0:
        flows = get_database_type_flows(database_project, database_name, db_type)
    elif ui2 == 1:
        flows = get_user_specific_flows(database_project, database_name)
    
    print('Chosen flows:')
    for f in flows:
        print(f)

    initialization = [database_project, database_name, flows, lcia_method, db_type]
    file_name_unique = f'{save_dir}\data_uniquie_{file_identifier}_{db_type}_{lcia_method}.xlsx'

    return flow_legend, database_name, file_name, sheet_name, save_dir, initialization, file_name_unique, db_type

def select_project_and_database():
    projects = bd.projects.report()
    proj_dct = {}
    for i, proj in enumerate(projects):
        proj_dct[proj[0]] = i

    proj_input = int(input(f'Select the given number for the project wished to use\n {proj_dct}'))

    chosen_proj = ''
    for key, item in proj_dct.items():
        if item == proj_input:
            # bd.projects.set_current(key)
            chosen_proj = key

    bd.projects.set_current(chosen_proj)

    database = bd.databases
    db_dct = {}

    for i, proj in enumerate(database):
        db_dct[proj] = i

    db_input = int(input(f'Select the given number for the database wished to use\n {db_dct}'))

    chosen_db = ''
    for key, item in db_dct.items():
        if item == db_input:
            # db = bd.Database(key)
            chosen_db = key

    print(f'The chosen project is {chosen_proj} and the chosen database is {chosen_db}')

    return chosen_proj, chosen_db

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
def get_database_type_flows(project_name: str, database: str, database_type: str):
    # Setting brightway to the given project
    bd.projects.set_current(project_name)

    # Specifiyng which database to use
    db = bd.Database(database)
    flows = []

    # For loop to extract the desired flows
    for act in db:
        if database_type in act['name']:
            flows.append(act['name'])
    
    flows.sort()
            
    # Returning the flows
    return flows

def get_user_specific_flows(project_name: str, database: str):
    # Setting brightway to the given project
    bd.projects.set_current(project_name)

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

# Setting up which databases to use depending on which project is worked in
def database_initialization(db_type, database_name, project_name):

    bd.projects.set_current(project_name)
    
    if 'biosphere3' in bd.databases:
        pass
    else:
        bi.bw2setup()

    eidb = bd.Database(database_name)
    
    if 'CONSQ' in db_type and 'sterilization' in project_name:
        db_eco = 'ev391consq'
        eidb_db = bd.Database(db_eco)
        all_acts = list(eidb) + list(eidb_db)

    elif 'APOS' in db_type and 'sterilization' in project_name:
        db_eco = 'ev391apos'

        eidb_db = bd.Database(db_eco)
        all_acts = list(eidb) + list(eidb_db)

    elif 'CONSQ' in db_type and 'ofir' in project_name.lower():
        db_eco = 'Consq EcoInvent'
        db_ofir = 'Ananas consq'

        eidb_ofir = bd.Database(db_ofir)
        eidb_db = bd.Database(db_eco)
        all_acts = list(eidb) + list(eidb_db) + list(eidb_ofir)

    elif 'APOS' in db_type and 'ofir' in project_name.lower():
        db_eco = 'APOS EcoInevnt'
        db_ofir = 'Ananas consq'

        eidb_ofir = bd.Database(db_ofir)
        eidb_db = bd.Database(db_eco)
        all_acts = list(eidb) + list(eidb_db) + list(eidb_ofir)

    else:
        db_eco = 'Consequential'
        db_cyl = 'Cylinder'
        db_pellet = 'Pellet'

        eidb_db = bd.Database(db_eco)
        eidb_cyl = bd.Database(db_cyl)
        eidb_pellet = bd.Database(db_pellet)

        all_acts = list(eidb) + list(eidb_db) + list(eidb_cyl) + list(eidb_pellet)

    return all_acts, eidb, eidb_db

# Function to initialize parameters for the LCIA calculations
def LCA_initialization(project_name: str, database_name: str, flows: list, method: str, db_type: str) -> tuple:
    all_acts, eidb, eidb_db = database_initialization(db_type, database_name, project_name)

    # Setting up an empty dictionary with the flows as the key
    procces_keys = {key: None for key in flows}

    size = len(flows)

    # Obtaining all the product codes for the process'
    for act in all_acts:
        for proc in range(size):
            if act['name'] == flows[proc]:
                procces_keys[flows[proc]] = act['code']

    process = []
    key_counter = 0

    # Obtaining all the subprocess in a list 
    for key, item in procces_keys.items():
        try:
            if eidb.get(item) in eidb:
                process.append(eidb.get(item))
            else:
                copied_process = copy_process(item, eidb_db, eidb)
                if copied_process:
                    process.append(copied_process)
                    print('Process copied')
                else:
                    print(f"Process with key '{item}' not found in the consequential database (eidb_consq) either.")
        except KeyError:
            print(f"Process with key '{item}' not found in the database '{eidb}'")
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



    FU = []
    # Creating the FU to calculate for
    for key, item in product_details.items():
        for idx in item:
            for n, m in idx.items():
                FU.append({key: {m[1]: m[0]}})

    print('Initialization is completed')
    return FU, impact_category, plot_x_axis

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
    df = df.set_axis(flow)

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
    categories, category_mapping = lp.category_organization(database_name)
    unique_elements = []
    for key, item in category_mapping.items():
        for ilst in item:
            unique_elements.append(ilst)

    return unique_elements

# Function to rearrange the data in the dataframe
def rearrange_dataframe_index(df):
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
            for i, plc in enumerate(plc_lst):
                # removing the chosen placement
                if user == plc:
                    plc_lst.remove(plc)

        # Creating the new index list
        idx_lst = [''] * len(idx_dct)
        for key, item in idx_dct.items():
            idx_lst[item] = key

        impact_category = df.columns
        df_rearranged = pd.DataFrame(0, index=idx_lst, columns=impact_category, dtype=object)

        row_counter = 0

        # Arranging the dataframe to the new dataframe
        for icol, col in enumerate(impact_category):
            for idx, row in df_rearranged.iterrows():
                rearranged_val = df.at[idx, col] # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html#pandas.DataFrame.at
                df_rearranged.iloc[row_counter, icol] = rearranged_val
                row_counter += 1
                
                if row_counter == len(idx_dct):  # Reset when all flows have been processed in the column
                    row_counter = 0
        return df_rearranged
    else:
        # If rearrnge is not chosen it returns the same dataframe as inputtet
        return df

def quick_LCIA_calculator(unique_process_index, func_unit, impact_categories, file_name_unique, sheet_name):
    if type(impact_categories) == tuple:
        impact_categories = [ic for ic in impact_categories]
    df_unique = pd.DataFrame(0, index=unique_process_index, columns=impact_categories, dtype=object)
    # unique_process_results = {}
    # calc_count = 1
    # total_calculations = len(uniquie_process_dct)*len(impact_categories)
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

def quick_LCIA(initialization, file_name, file_name_unique, sheet_name):
    database_project, database_name, flows, lcia_method, db_type = initialization
    functional_unit, impact_category, plot_x_axis_all = LCA_initialization(database_project, database_name, flows, lcia_method, db_type)


    # Ensure impact categories is a list
    impact_categories = list(impact_category) if isinstance(impact_category, tuple) else impact_category
    # Loop through each impact category and flow
    
    unique_process_index = []
    uniquie_process = []

    # for f in flows:    
        # Find matches in functional units and calculate LCA
    for func_dict in functional_unit:
        for FU_key, FU_item in func_dict.items():
            # if f in FU_key:
            for proc in FU_item.keys():
                if f'{proc}' not in unique_process_index:
                    unique_process_index.append(f'{proc}')
                    uniquie_process.append(proc)

    unique_process_index.sort()


    FU = []
    for upi in unique_process_index:
        for proc in uniquie_process:
            if upi == f'{proc}':
                FU.append({proc : 1})

    user_input = ''
    # quick_LCIA_calculator(unique_process_index, FU, impact_categories, file_name_unique, sheet_name)

    # Check if file exists
    if os.path.isfile(file_name_unique): # https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
        # Import LCIA results
        try:
            df_unique = import_LCIA_results(file_name_unique, unique_process_index, impact_category)
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

    if 'n' not in user_input.lower():
        df_unique_new = import_LCIA_results(file_name_unique, unique_process_index, impact_category)

        df = pd.DataFrame(0, index=flows, columns=impact_categories, dtype=object)

        for col in impact_categories:
            for i, row in df.iterrows():
                row[col] = []

        for col, impact in enumerate(impact_category):
            for fu in functional_unit:
                for key, item in fu.items():
                    proc = str([p for p in item.keys()][0])
                    val = float([v for v in item.values()][0])
                    factor = df_unique_new.at[proc, impact]
                    impact_value = val * factor
                    
                    # print(key, proc, val, factor, impact_value, val*factor == impact_value)
                    df.at[key, impact].append([proc, impact_value])
        save_LCIA_results(df, file_name, sheet_name, impact_categories)

    
    df = import_LCIA_results(file_name, flows, impact_category)

    return df, plot_x_axis_all, impact_categories