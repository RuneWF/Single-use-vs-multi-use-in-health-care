# Import packages
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

from  standards import *
import life_cycle_assessment as lc
importlib.reload(lc)

def flow_name_update(x, gwp, db_type, database_name):
    if 'Ananas consq' in database_name or 'sterilization' in database_name:
        # print(x, gwp)
        x_og = x
        if f'- {db_type}' in x:
            #print(key)
            x = x.replace(f' - {db_type}', '')
        if 'alubox' in x:           
            x = x.replace('alubox ', '')
            if ('raw' in x and 'avoid' in x) or ('raw' in x and gwp < 0):
                x = 'Avoided mat. prod.'
            elif 'raw' in x and 'avoid' not in x.lower():
                x = x.replace('raw materials', 'Raw mat.')
        
            if 'production' in x:
                x = 'Manufacturing'
            if 'EoL' in x:
                x = 'Avoided mat. prod.'
                if gwp > 0:
                    # print(x_og, gwp)
                    gwp = -gwp
                    # print(x_og, gwp)
        if 'sheet manufacturing' in x:
            x = 'Manufacturing'
        if 'electricity' in x:
            x = 'Avoided energy prod.'
        if 'heating' in x:
            x = 'Avoided energy prod.'
        if 'market for polypropylene' in x:
            if gwp < 0:
                x = 'Avoided mat. prod.'
            else:
                x = 'Raw mat.'
        if 'PE granulate' in x:
            if gwp < 0:
                x = 'Avoided mat. prod.'
            else:
                x = 'Raw mat.'
        if 'no Energy Recovery' in x:
            x = 'Incineration'#x.replace(' no Energy Recovery', '')

        if 'board box' in x or 'packaging film' in x:
            x = 'Packaging'
        # if 'pp' in x:
        #     x = x.replace('pp', 'PP')
        if 'autoclave' in x:
            x = x.replace('autoclave', 'Autoclave')
        if 'transport' in x:
            x = 'Transport'
        if 'cabinet' in x or 'wipe' in x:
            x = 'Box cleaning'
        if 'polysulfone' in x:
            x = 'Manufacturing'

    elif 'Lobster' in database_name:
            if 'sc1' in x:
                x = x.replace(f'sc1 ', '')
            elif 'sc2' in x:
                x = x.replace(f'sc2 ', '')
            elif 'sc3' in x:
                x = x.replace(f'sc3 ', '')
            elif 'Waste' in x:
                x = 'Incineration'
            elif 'market for electricity' in x:
                x = 'Avoided electricity'
            elif 'heating' in x:
                x = 'Avoided heat'
            elif 'erbe' in x:
                x = 'Erbe'
            elif '(use) DK' in x:
                x = x.replace(' (use) DK', '')
            elif '(use) - DK' in x:
                x = x.replace(' (use) - DK', '')
            elif '(prod) DE' in x:
                x = x.replace(' (prod) DE', '')
            elif 'Remanufacturing' in x:
                x = 'Remanufacturing'

    return x


def scaled_FU_plot(df_scaled, plot_x_axis, inputs, impact_category, legend_position):
    import os

    
    flow_legend = inputs[0]
    colors = inputs[1]
    save_dir = inputs[2]
    db_type = inputs[3]

    columns_to_plot = df_scaled.columns

    index_list = list(df_scaled.index.values)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 1/(len(index_list) + 1) 
    index = np.arange(len(columns_to_plot))

    # Plotting each group of bars
    min_val = 0
    for i, process in enumerate(df_scaled.index):
        values = df_scaled.loc[process, columns_to_plot].values
        ax.bar((index + i * bar_width), values, bar_width, label=process, color=colors[i])  
        # print(min(values))
        if min_val > min(values):
            min_val = min(values)


    # Setting labels and title
    ax.set_title(f'Scaled impact of the Functional Unit - {impact_category[0][0]}',weight='bold',fontsize=16)
    ax.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax.set_xticklabels(plot_x_axis,  weight='bold', fontsize=12)

    if 'endpoint' not in impact_category[0][0]:
        plt.xticks(rotation=90)
    else: 
        plt.xticks(rotation=0)

    # plt.yticks(np.arange(-1.6, 1.01, step=0.2))
    # plt.ylim(-1.62,1.03)

    # fig.(weight='bold', fontsize=12)

    ax.legend(flow_legend,bbox_to_anchor=(1.01, legend_position, .1, 0), loc="lower left",
                mode="expand", borderaxespad=0,  ncol=1, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'scaled_impact_score_multi_{db_type}_{impact_category[0][0]}.jpg'), bbox_inches='tight')
    plt.show()

def single_score_plot(directory, df_tot, inputs):

    flow_legend = inputs[0]
    colors = inputs[1]
    save_dir = inputs[2]
    db_type = inputs[3]

    lst_scaled = lc.LCIA_normalization(directory, df_tot)

    index_list = list(df_tot.index.values)
    # Plotting
    fig, ax = plt.subplots(figsize=(9,7))
    bar_width = 1/(len(index_list)) 
    index = np.arange(len(index_list))   

    ax.bar(index + bar_width, lst_scaled, bar_width, label=index_list, color=colors)

    # Setting labels and title
    ax.set_title('Scaled single score of the Functional Unit',weight='bold')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(flow_legend)
    #plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 1.01, step=0.1))
    plt.ylim(0,1.03)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'scaled_single_score_multi_{db_type}.jpg'), bbox_inches='tight')
    plt.show()

def gwp_lc_plot(df_GWP, category_mapping, categories, inputs, y_axis_values):
    import os
    
    flow_legend = inputs[0]
    colors = inputs[1]
    save_dir = inputs[2]
    db_type = inputs[3]
    database_name = inputs[4]

    x_axis = []
    GWP_value = []

    for col in df_GWP.columns:
        for i, row in df_GWP.iterrows():
            lst_x = []
            lst_GWP = []
            gwp_tot = 0
            for lst in row[col]:
                x = lst[0]
                gwp = lst[1]
                # print(lst)
                #print(gwp,x)

                x = flow_name_update(x, gwp, db_type, database_name)

                lst_x.append(x)
                lst_GWP.append(gwp)
                gwp_tot += gwp
                # print(x, gwp)
            # print(gwp_tot, lst_GWP)
            lst_GWP.append(gwp_tot)
            lst_x.append('Total')
            x_axis.append(lst_x)
            GWP_value.append(lst_GWP)

    # Ensure the legend displays items in the category order
    ordered_legend = {key: [] for key in category_mapping}

    for x_lst in range(len(x_axis)):
        for x in range(len(x_axis[x_lst])):
            
            for key, item in category_mapping.items():
                    # print(x_axis[x_lst][x], item, x_axis[x_lst][x] in item)
                    if x_axis[x_lst][x] in item:
                        # print(x_axis[x_lst][x], item, x_lst, x)
                        ordered_legend[key].append(x_axis[x_lst][x])

    plot_legend = {key: [] for key in category_mapping}
    temp = []

    for key,value in ordered_legend.items():
        #print(key, value)
        for val in value:
            if val not in temp:
                temp.append(val)
                # print(val)
                plot_legend[key].append(val)

    color_map = {}
    #unique_processes = {process for sublist in x_axis for process in sublist}
    for i, process in enumerate(temp):
        color_map[process] = colors[i]
        #print(process, i)



    # Initialize an ordered dictionary for legend_handles to maintain the order
    legend_handles = OrderedDict()

    # Initialize legend_handles with keys from plot_legend and empty lists
    for process in temp:
        legend_handles[process] = None

    # Plotting logic
    if len(x_axis) == len(GWP_value):
        num_scenarios = len(GWP_value)  # Number of scenarios
        bar_width = 0.15  # Width of the bars for each scenario
        space_between_scenarios = 0.05  # Space between each scenario set
        index = np.arange(len(categories))  # X-axis index positions for the categories

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        idx = ['x', '^', 'o', 'D', 'P', '2', '*', 'H']  # List of markers for scenarios

        all_markers = []  # List to store Line2D objects for markers

        # Main plotting logic
        for scenario in range(num_scenarios):
            bottom_positive = np.zeros(len(categories))  # Initialize the bottom array for positive values
            bottom_negative = np.zeros(len(categories))  # Initialize the bottom array for negative values
            scenario_index = index + scenario * (bar_width + space_between_scenarios)

            for length in range(len(x_axis[scenario])):
                process_name = x_axis[scenario][length]
                value = GWP_value[scenario][length]

                # Determine which category this process falls into
                for i, category in enumerate(categories):
                    if any(keyword in process_name for keyword in category_mapping[category]):
                        # Assign color based on the process name
                        color = color_map[process_name]

                        # Create a bar with the specific color
                        if value >= 0:
                            bar = ax.bar(scenario_index[i], value, bar_width,
                                        label=f"{process_name}" if legend_handles[process_name] is None else "",
                                        bottom=bottom_positive[i],
                                        color=color)
                            bottom_positive[i] += value
                        else:
                            bar = ax.bar(scenario_index[i], value, bar_width,
                                        label=f"{process_name}" if legend_handles[process_name] is None else "",
                                        bottom=bottom_negative[i],
                                        color=color)
                            bottom_negative[i] += value

                        # Add the bar to the corresponding process in legend_handles
                        if legend_handles[process_name] is None:
                            legend_handles[process_name] = bar

                        # Add plot markers (symbols) at the bottom
                        ax.plot(scenario_index[i], y_axis_values[0]+0.1, marker=idx[scenario], color='gray')

                        break

        # Add custom markers to the legend
        for i, marker in enumerate(idx):
            all_markers.append(Line2D([0], [0], marker=marker, color='gray', label=f'Scenario {i + 1}', linestyle='None'))

        # Set x-axis labels and ticks, adjusting to account for spacing
        tick_positions = index + (num_scenarios - 1) * (bar_width + space_between_scenarios) / 2
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(categories)

        # Axis limits
        ax.set_ylim(y_axis_values[0]- 0.03, y_axis_values[1] + 0.05)
        ax.set_yticks(np.arange(y_axis_values[0], y_axis_values[1]+0.01, step=y_axis_values[2]))
        ax.set_ylabel("Global Warming Potential [kg CO$_2$e]")
        ax.set_title(f'GWP impact for each life stage for 1 FU - {db_type}')

        # Add markers to legend_handles for display at the bottom of the legend
        valid_legend_handles = [(k, v) for k, v in legend_handles.items() if v is not None]
        legend_handles_for_display = valid_legend_handles + [(flow_legend[i], marker) for i, marker in enumerate(all_markers)]

        if legend_handles_for_display:
            ax.legend(handles=[v for k, v in legend_handles_for_display], labels=[k for k, v in legend_handles_for_display], bbox_to_anchor=(1.005, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'GWP_life_stage_{db_type}.jpg'), bbox_inches='tight')
        plt.show()
        

    else:
        print('The x-axis and GWP values have different sizes')

def category_organization(database_name):
    categories = ["Raw mat. + prod.", "Use", "Transport", "EoL", "Total"]

    if 'Ananas consq' in database_name or 'sterilization' in database_name:
        category_mapping = {
        "Raw mat. + prod.": ["Raw mat.", "Manufacturing", "Packaging"],
        "Use": ["Autoclave", "Box cleaning"],
        "Transport": ["Transport"],
        "EoL": ["Incineration", "Avoided mat. prod.", "Avoided energy prod."],
        "Total": ["Total"]
        }
    
    elif 'Lobster' in database_name:
        category_mapping = {
        "Raw mat. + prod.": ["Diathermy", "Bipolar burner", "Scalpel"],
        "Use": ["Autoclave", "Dishwasher", "Erbe", "Remanufacturing"],
        "Transport": ["Transport"],
        "EoL": ["Incineration", "Avoided heat", "Avoided electricity"],
        "Total": ["Total"]
        }

    return categories, category_mapping

def gwp_scenario_plot(df_GWP, inputs):
    flow_legend = inputs[0]
    colors = inputs[1]
    save_dir = inputs[2]
    db_type = inputs[3]
    database_name = inputs[4]

    categories, category_mapping = category_organization(database_name)
    columns = []
    for key, item in category_mapping.items():
        for ilst in item:
            columns.append(ilst)


    x_axis = []
    GWP_value = []

    for col in df_GWP.columns:
        for i, row in df_GWP.iterrows():
            lst_x = []
            lst_GWP = []
            gwp_tot = 0
            for lst in row[col]:
                x = lst[0]
                gwp = lst[1]
                # print(lst)
                # print(gwp,x)

                x = flow_name_update(x, gwp, db_type, database_name)
                if 'Avoided mat. prod.' in x and gwp > 0:
                    gwp = -gwp

                lst_x.append(x)
                lst_GWP.append(gwp)
                gwp_tot += gwp
            
            lst_GWP.append(gwp_tot)
            lst_x.append('Total')
            x_axis.append(lst_x)
            GWP_value.append(lst_GWP)


    # Create an empty dictionary to collect the data
    test_df = {}

    # Collect the data into the dictionary
    for i, p in enumerate(GWP_value):
        for a, b in enumerate(p):
            key = (flow_legend[i], x_axis[i][a])
            if key in test_df:
                test_df[key] += b
            else:
                test_df[key] = b

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(list(test_df.items()), columns=['Category', 'Value'])

    # Separate 'Total' values from the rest
    totals_df = df[df['Category'].apply(lambda x: x[1]) == 'Total']
    df = df[df['Category'].apply(lambda x: x[1]) != 'Total']

    # Pivot the DataFrame to have a stacked format
    df_stacked = df.pivot_table(index=[df['Category'].apply(lambda x: x[0])], columns=[df['Category'].apply(lambda x: x[1])], values='Value', aggfunc='sum').fillna(0)

    # Create a DataFrame to store results
    df_stack_updated = pd.DataFrame(0, index=flow_legend, columns=columns[:-1], dtype=object)  # dtype=object to handle lists
    for col in df_stack_updated.columns:
        for inx, row in df_stack_updated.iterrows():
            row[col] = df_stacked[col][inx]


    # Plotting the stacked bar chart
    ax = df_stack_updated.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)

    # Plotting 'Total' values as dots and including it in the legend
    for flow in flow_legend:
        # print(flow)
        for idx, row in totals_df.iterrows():
            # print( row['Category'][0])
            if flow in row['Category'][0]:
                unit = row['Category'][0]
                total = row['Value']
                # print(unit, total)
                ax.plot(unit, total, '^', color='k', label='Total' if idx == 0 else "")

    # Custom legend with 'Total' included
    handles, labels = ax.get_legend_handles_labels()

    handles.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=10, label='Total'))
    ax.legend(labels=columns, handles=handles, bbox_to_anchor=(1.01, .591, .23, 0), loc="lower left", mode="expand", borderaxespad=0, ncol=1, fontsize=10)

    # Setting labels and title
    plt.title(f'GWP impact for each life stage for 1 FU - {db_type}', weight='bold')
    plt.ylabel('Global Warming Potential [kg CO$_2$e]', weight='bold')
    plt.ylim(-.85,1.65)
    plt.yticks(np.arange(-.8, 1.6, step=0.2))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'_GWP_life_stage_pr_scenario_{db_type}.jpg'), bbox_inches='tight')
    plt.show()

