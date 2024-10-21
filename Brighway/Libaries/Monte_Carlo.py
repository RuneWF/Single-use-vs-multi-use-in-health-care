# Import packages we'll need later on in this tutorial
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

# Import BW25 packages
import bw2data as bd
import bw2io as bi
import bw2calc as bc
import bw2analyzer as bwa
import brightway2 as bw 
from bw2calc import LeastSquaresLCA

from standards import *

save_dir = results_folder('Results')

def MC_set_up(flows, database):
    eidb = bd.Database(database)

    FU_new = {}
    FU_uncertainties = {}
    for flow in flows:
        for act in eidb:
            
            # for sc in range(1,4):
            if flow in act['name']:
                # print(act['name'])
                FU_new[flow] = {}
                FU_uncertainties[flow] = {}
                # print(act['name'])
                for exc in act.exchanges():
                    # print(exc.input, exc['type'])
                    if exc['type'] == 'technosphere':
                        # print(exc.input, exc.input['name'], exc['amount'], exc['type'])
                        FU_new[flow].update({exc.input : exc['amount']})
                        FU_uncertainties[flow].update({exc.input : exc.uncertainty})
    return FU_new, FU_uncertainties

def sample_lognormal(loc, scale):
    # Convert loc and scale to the parameters of the lognormal distribution
    mean = np.exp(loc)
    sigma = scale
    return np.random.lognormal(mean, sigma)

def data_sample(FU, uncertainties):
    sampled_data = {}

    for uncertainty_key, uncert in uncertainties.items():
        for uk, ui in uncert.items():
            if uk == 'uncertainty type' and ui == 2:  # Lognormal distribution
                sampled_data[uncertainty_key] = sample_lognormal(uncert['loc'], uncert['scale'])

    FU_copy = copy.deepcopy(FU)

    for key_copy, item_copy in FU.items():
        # print(FU_new['sc1 - No DU'][k_new])
        if key_copy in sampled_data.keys():
            updated_value = sampled_data[key_copy] * item_copy
            FU_copy[key_copy] = updated_value
            # print(k_new, sampled_data[k_new], i_new)


    return FU_copy

def MonteCarlo(iterations, FU, impact_category, uncertainties):
    # Initialize an array to store the LCIA results for each iteration
    lcia_results_array = np.zeros(iterations)

    # Perform Monte Carlo simulation
    for i in range(iterations):

        
        FU_updated = data_sample(FU, uncertainties)

        # Use the sampled data in the Monte Carlo LCA
        MC_lca = bw.MonteCarloLCA(FU_updated, impact_category)
        MC_lca.lci()
        
        # Initialize cf_params if not already set
        if not hasattr(MC_lca, 'cf_params'):
            MC_lca.cf_params = MC_lca.load_lcia_data()

        # Rebuild the characterization matrix if it's not already initialized
        if not hasattr(MC_lca, 'characterization_matrix'):
            MC_lca.rebuild_characterization_matrix(MC_lca.method)

        # Perform LCIA calculation directly
        MC_lca.lcia_calculation()
        
        # Store the LCIA result in the array
        lcia_results_array[i] = MC_lca.score
        print(f'Iteration {i+1} of {iterations}')
    
    return lcia_results_array

# Create a function to clean the sheet names
def clean_sheet_name(sheet_name, max_length=31):
    # Replace invalid characters with an underscore
    cleaned_name = re.sub(r'[\\/:*?\'<>\[\]]', ' -', sheet_name)
    # Truncate the name to the max allowable length (31 characters)
    return cleaned_name[:max_length]

def MC_simulation(iterations, flows, database, impact_category, results_file):
    FU_new, uncertainties = MC_set_up(flows, database)
    MC_idx = ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum']

    if type(impact_category) == tuple:        

        # Define the LCIA method
        # for method in impact_category:
        print(f'Doing Monte Carlo simulations for {impact_category}')

        # Create a DataFrame to store results
        df_MC = pd.DataFrame(0, index=MC_idx, columns=FU_new.keys(), dtype=object)  # dtype=object to handle lists
        raw_data = []
        for col in df_MC.columns:
            print(f'Processing {col}')
            lcia_results_array = MonteCarlo(iterations, FU_new[col], impact_category, uncertainties[col])
            raw_data.append(lcia_results_array)
            for index, row in df_MC.iterrows():
                # print(index, row[col])
                if 'mean' in index.lower():
                    row[col] = np.mean(lcia_results_array)
                    # print(index)
                elif 'median' in index.lower():
                    row[col] = np.median(lcia_results_array)
                    # print(index)
                elif 'standard' in index.lower():
                    row[col] = np.std(lcia_results_array)
                    # print(index)
                elif 'minimum' in index.lower():
                    row[col] = np.min(lcia_results_array)
                    # print(index)
                elif 'maximum' in index.lower():
                    row[col] = np.max(lcia_results_array)
            with pd.ExcelWriter(f'{impact_category[1]} -' + results_file, engine='xlsxwriter') as writer:
                # Get the sheet name and clean it
                sheet_name = clean_sheet_name(impact_category[1])
                # Save the dataframe to the corresponding sheet
                df_MC.to_excel(writer, sheet_name=sheet_name)

        return df_MC, raw_data
        
    else:
        Monte_Carlo_dct = {}

        for method in impact_category:
            print(f'Doing Monte Carlo simulations for {method}')

            # Create a DataFrame to store results
            df_MC = pd.DataFrame(0, index=MC_idx, columns=FU_new.keys(), dtype=object)  # dtype=object to handle lists
            raw_data = []
            for col in df_MC.columns:
                print(f'Processing {col}')
                lcia_results_array = MonteCarlo(iterations, FU_new[col], method, uncertainties[col])
                raw_data.append(lcia_results_array)
                for index, row in df_MC.iterrows():
                    # print(index, row[col])
                    if 'mean' in index.lower():
                        row[col] = np.mean(lcia_results_array)
                        # print(index)
                    elif 'median' in index.lower():
                        row[col] = np.median(lcia_results_array)
                        # print(index)
                    elif 'standard' in index.lower():
                        row[col] = np.std(lcia_results_array)
                        # print(index)
                    elif 'minimum' in index.lower():
                        row[col] = np.min(lcia_results_array)
                        # print(index)
                    elif 'maximum' in index.lower():
                        row[col] = np.max(lcia_results_array)
                        # print(index)
                    
                    
                        
            Monte_Carlo_dct[method[1]] = df_MC

            with pd.ExcelWriter(results_file, engine='xlsxwriter') as writer:
                for i, dataframe in enumerate(Monte_Carlo_dct.values()):
                    # Get the sheet name and clean it
                    sheet_name = clean_sheet_name(impact_category[i][1])
                    # Save the dataframe to the corresponding sheet
                    dataframe.to_excel(writer, sheet_name=sheet_name)

        return Monte_Carlo_dct, raw_data

def MC_graphs(flow_legend, data, iterations):
    
    for scenario, data_point in enumerate(data):
        data_point = data_point.tolist()
        # Create box plot for each scenario (column)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data_point)

        # Set plot labels and title
        plt.title(f'Box Plots for sc. {scenario+1} - {flow_legend[scenario]} - Iterations = {iterations}', size=16)
        plt.xlabel(flow_legend[scenario], size=12)
        plt.ylabel('kg CO$_2$e', size=12)


        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'MC_results_{flow_legend[scenario]}.jpg'), bbox_inches='tight')
        # Show the plot
        plt.show()



                