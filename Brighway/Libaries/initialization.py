import pandas as pd
import json
import copy

import ipywidgets as widgets
from IPython.display import display
import logging
import concurrent.futures
import multiprocessing as mp

# Import BW25 packages
import bw2data as bd
import bw2io as bi
import brightway2 as bw 
from bw2calc import bc

# Importing self-made libaries
from standards import *
import LCA_plots as lp

# Updated function to select project and database with widgets
def select_project_and_database():
    projects = bd.projects.report()
    proj_dct = {proj[0]: i for i, proj in enumerate(projects)}

    # Create a dropdown for project selection
    project_dropdown = widgets.Dropdown(
        options=proj_dct,
        description='Project:'
    )
    display(project_dropdown)

    # Wait until the project is selected
    selected_proj = None
    def on_project_change(change):
        nonlocal selected_proj
        selected_proj = change.new
        bd.projects.set_current(selected_proj)
        print(f'Selected Project: {selected_proj}')

    project_dropdown.observe(on_project_change, names='value')

    # Now, let's handle database selection
    database = bd.databases
    db_dct = {db: i for i, db in enumerate(database)}

    # Create a dropdown for database selection
    db_dropdown = widgets.Dropdown(
        options=db_dct,
        description='Database:'
    )
    display(db_dropdown)

    # Handle database selection
    selected_db = None
    def on_db_change(change):
        nonlocal selected_db
        selected_db = change.new
        print(f'Selected Database: {selected_db}')

    db_dropdown.observe(on_db_change, names='value')

    return selected_proj, selected_db

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


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Example of adding logging to your LCA initialization process
def LCA_initialization(project_name: str, database_name: str, flows: list, method: str, db_type: str) -> tuple:
    logging.debug(f'Initializing LCA for project: {project_name}, database: {database_name}, method: {method}')

    all_acts, eidb, eidb_db = database_initialization(db_type, database_name, project_name)
    logging.debug(f'Database initialized, found {len(all_acts)} activities.')

    procces_keys = {key: None for key in flows}
    size = len(flows)

    # Additional logging when mapping flows to processes
    for act in all_acts:
        for proc in range(size):
            if act['name'] == flows[proc]:
                procces_keys[flows[proc]] = act['code']
                logging.debug(f'Mapped flow "{flows[proc]}" to process code "{act["code"]}"')


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
    return FU, FU_procces, impact_category, plot_x_axis

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



def save_LCIA_results(df, file_name, sheet_name, impact_category):
    try:
        if type(impact_category) == tuple:
            impact_category = [impact_category]

        # Convert each cell to a JSON string for all columns
        df_save = df.map(lambda x: json.dumps(x) if isinstance(x, list) else x)

        # Save to Excel
        with pd.ExcelWriter(file_name) as writer:
            df_save.to_excel(writer, sheet_name=sheet_name, index=False, header=True)

        logging.info('DataFrame with nested lists written to Excel successfully.')

        with open("impact_categories", "w") as fp:
            json.dump(impact_category, fp)

    except Exception as e:
        logging.error(f'Error saving LCIA results: {e}')
        raise

def import_LCIA_results(file_name, flow, impact_category):
    try:
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

        logging.info(f'LCIA results imported from {file_name}')
        return df
    except Exception as e:
        logging.error(f'Error importing LCIA results: {e}')
        raise

def lca_worker(args):
    logging.debug(f'Worker received args: {args}')
    try:
        proc, factor, impact = args  # Expecting exactly three elements here
        # Your existing processing logic here
    except ValueError as e:
        logging.error(f'ValueError: {e} - Arguments: {args}')
        raise  # Re-raise the exception for further handling

def run_lcia_in_threads(tasks):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lca_worker, tasks))
    return results

# Step 6: Run the LCIA calculation using multiprocessing
def quick_LCIA_calculator(processes, impact_categories, num_cores=4):
    tasks = [(proc, 1, impact) for proc in processes for impact in impact_categories]
    
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(lca_worker, tasks)

    # Aggregate results into a DataFrame
    df = pd.DataFrame(0, index=[r[0] for r in results], columns=impact_categories)
    for proc, impact, score in results:
        if score is not None:
            df.at[proc, impact] = score
    return df