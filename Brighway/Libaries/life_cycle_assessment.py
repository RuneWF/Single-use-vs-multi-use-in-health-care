import pandas as pd
import json
import copy

# Import BW25 packages
import bw2data as bd
import bw2io as bi
import brightway2 as bw 

# Importing self-made libaries
from standards import *
import LCA_plots as lp


# Function to obtain the LCIA category to calculate the LCIA results
def lcia_method(method):
    # Checking if the LCIA method is ReCiPe, and ignores difference between lower and upper case 
    if 'recipe' in method.lower():
        # Using H (hierachly) due to it has a 100 year span
        # Obtaining the midpoint categpries and ignoring land transformation (Land use still included)
        all_methods = [m for m in bw.methods if 'ReCiPe 2016 v1.03, midpoint (H)' in str(m) and 'no LT' not in str(m)] # Midpoint

        # Obtaining the endpoint categories and ignoring land transformation
        endpoint = [m for m in bw.methods if 'ReCiPe 2016 v1.03, endpoint (H)' in str(m) and 'no LT' not in str(m) and 'total' in str(m)]

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
def get_flows(project_name: str, database: str, database_type: str):
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

    FU_procces = []

    # Obtaining the process for the FU
    for flow in flows:
        for flow_length in range(len(product_details[flow])):
            for key in product_details[flow][flow_length].keys():
                if flow in key:
                    key = key.replace(f'{flow} ', '')
                FU_procces.append(key)

    FU = []
    # Creating the FU to calculate for
    for key, item in product_details.items():
        for idx in item:
            for n, m in idx.items():
                FU.append({key: {m[1]: m[0]}})

    print('Initialization is completed')
    return FU, FU_procces, impact_category, plot_x_axis #, product_details_code

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

# Function to perform the LCIA calculations
def life_cycle_impact_assessment(flows, functional_unit, impact_categories, process):
    print()
    print('Calculating the LCA results:')

    # Checking if the impact category is a tuple, and if true it is turned into a list
    if type(impact_categories) == tuple:
        impact_categories = [impact_categories]

    # Define the amount of columns
    m = len(impact_categories)

    # Create a DataFrame to store results
    df = pd.DataFrame(0, index=flows, columns=impact_categories, dtype=object)  # dtype=object to handle lists

    calc_count = 1
    row_counter = 0

    # Loop through each impact categories to calculate LCIA results for each process
    for col, impact in enumerate(impact_categories):
        # Loop through flows
        for f in flows:
            df_lst = []  # Clear list for each flow in each impact category
            for func_unit in range(len(functional_unit)):
                for FU_key, FU_item in functional_unit[func_unit].items():
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

            
            if row_counter == len(flows):  # Reset when all flows have been processed in the column
                row_counter = 0
    
    # Returning the dataframe
    return df

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

# Simplified function to perform the initialization and LCIA results 
def calculate_lcia(calculate, initialization, file_name, sheet_name):

    database_project, database_name, flows, lcia_meth, db_type = initialization
    # Performing the initialization
    FU, FU_proc, impact_category, plot_x_axis_all = LCA_initialization(database_project, database_name, flows, lcia_meth, db_type)
    
    # Checking if the LCIA calculations needs to be performed  
    if calculate is True:
        # Caculating the LCA results
        df = life_cycle_impact_assessment(flows, FU, impact_category, FU_proc)
        save_LCIA_results(df, file_name, sheet_name, impact_category)

    # Import the LCIA results
    df = import_LCIA_results(file_name, flows, impact_category)

    # Return the dataframe with LCIA results, the LCIA categories and the names for each value on the x-axis
    return df, impact_category, plot_x_axis_all

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
