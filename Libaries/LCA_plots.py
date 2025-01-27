import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from matplotlib.lines import Line2D  # Import for creating custom legend markers
import importlib
import os
from copy import deepcopy as dc

from  standards import *
import life_cycle_assessment as lc
importlib.reload(lc)

# Function to update the flow name and simplify them
def flow_name_update(x, gwp, db_type, database_name):
    x_og = x
    # x_lst = []
    # if x not in x_lst:
    #     x_lst.append(x)
    #     print(x)

    if 'case1' in database_name:
        
        if 'H200' in x or 'H400' in x or 'alubox (small)' in x or 'alubox (large)' in x:
            # print(x)
            x = 'Raw mat. + prod.' 
        if 'market for polypropylene' in x or 'polyethylene, high density' in x and 'waste' not in x:
            # print(f'Plastic {x} = {gwp}')
            x = "Avoided mat. prod."
        if 'recyc' in x.lower() or 'aluminium scrap' in x:
            # print(f'recycling {x} = {gwp}')
            # print(x, gwp)
            x = 'Recycling'
        if 'waste paper' in x:
            # print(f'paper {x} = {gwp}')
            x = 'Recycling'
        if 'electricity' in x or 'high voltage' in x or 'heating' in x:
            x = 'Avoided energy prod.'
        if 'incineration' in x or 'waste' in x:
            x = 'Incineration'
        if 'board box' in x or 'packaging film' in x:
            x = 'Raw mat. + prod.'
        if 'autoclave' in x:
            x = 'Autoclave'
        if 'transport' in x:
            x = 'Raw mat. + prod.'
        if 'polysulfone' in x:
            x = 'Raw mat. + prod.'
        if 'cast alloy' in x:
            # print(f'Cast alloy {x} = {gwp}')
            x = "Avoided mat. prod."
        if 'wipe' in x or 'mechanical disinfection' in x:
            x = 'Disinfection'


    elif 'case2' in database_name or 'model' in database_name:
        if 'H200' in x:
            print(x, gwp)
            x = 'Ster. consumables' 
        if 'autoclave' in x.lower():
            x = 'Ster. autoclave' 
        if ('MUD' in x or 'SUD' in x) and 'eol' not in x or 'transport' in x or 'scalpel' in x:
            x = "Raw mat. + prod."
        if 'recyc' in x.lower() or 'aluminium scrap' in x:
            # print(x, gwp)
            x = 'Recycling'
        if 'waste paper' in x:
            x = 'Avoided mat. prod.'
        if 'electricity' in x or 'high voltage' in x or 'heating' in x:
            x = 'Avoided energy prod.'
        if 'incineration' in x or 'waste' in x:
            x = 'Incineration'
        if 'board box' in x or 'packaging film' in x:
            x = 'Raw mat. + prod.'

        if 'wipe' in x or 'mechanical disinfection' in x:
            x = 'Disinfection'
        if 'eol' in x:
            x = 'Incineration'
        if 'use' in x:
            x = 'Use'
          

    return x, gwp

def break_even_flow_seperation(x, gwp, db_type, database_name):
    x_og = x
    # x_lst = []
    # if x not in x_lst:
    #     x_lst.append(x)
    #     print(x)

    if 'case1' in database_name:
        if 'H200' in x or 'H400' in x or 'alubox (small)' in x or 'alubox (large)' in x:
            # print(x)
            x = 'Raw mat. + prod.' 
        if 'market for polypropylene' in x or 'polyethylene, high density' in x and 'waste' not in x:
            # print(f'Plastic {x} = {gwp}')
            x = "Avoided mat. prod."
        if 'recyc' in x.lower() or 'aluminium scrap' in x:
            # print(f'recycling {x} = {gwp}')
            # print(x, gwp)
            x = 'Recycling'
        if 'waste paper' in x:
            # print(f'paper {x} = {gwp}')
            x = 'Recycling'
        if 'electricity' in x or 'high voltage' in x or 'heating' in x:
            x = 'Avoided energy prod.'
        if 'incineration' in x or 'waste' in x:
            x = 'Incineration'
        if 'board box' in x or 'packaging film' in x:
            x = 'Raw mat. + prod.'
        if 'autoclave' in x:
            x = 'Autoclave'
        if 'transport' in x:
            x = 'Raw mat. + prod.'
        if 'polysulfone' in x:
            x = 'Raw mat. + prod.'
        if 'cast alloy' in x:
            # print(f'Cast alloy {x} = {gwp}')
            x = "Avoided mat. prod."
        if 'wipe' in x or 'mechanical disinfection' in x:
            x = 'Disinfection'


    elif 'case2' in database_name or 'model' in database_name:
        if 'H200' in x:
            print(x, gwp)
            x = 'Ster. consumables' 
        if 'autoclave' in x.lower():
            x = 'Ster. autoclave'
        if ('MUD' in x or 'SUD' in x) and 'eol' not in x or 'transport' in x or 'scalpel' in x:
            x = "Raw mat. + prod."
        if 'recyc' in x.lower() or 'aluminium scrap' in x:
            # print(x, gwp)
            x = 'Recycling'
        if 'waste paper' in x:
            x = 'Avoided mat. prod.'
        if 'electricity' in x or 'high voltage' in x or 'heating' in x:
            x = 'Avoided energy prod.'
        if 'incineration' in x or 'waste' in x:
            x = 'Incineration'
        if 'board box' in x or 'packaging film' in x:
            x = 'Raw mat. + prod.'

        if 'wipe' in x or 'mechanical disinfection' in x:
            x = 'Disinfection'
        if 'eol' in x:
            x = 'Incineration'
        if 'use' in x:
            x = 'Use'
          

    return x, gwp

# Function for single score plot for EF LCIA results
def single_score_plot(directory, df_tot, inputs):

    flow_legend = inputs[0]
    colors = inputs[1]
    save_dir = inputs[2]
    db_type = inputs[3]

    # Creating an list where each column is
    lst_scaled = lc.LCIA_normalization(directory, df_tot)

    index_list = list(df_tot.index.values)
    # Plotting
    fig, ax = plt.subplots(figsize=(9,7))
    bar_width = 1/(len(index_list)) 
    index = np.arange(len(index_list))   

    ax.bar(index + bar_width, lst_scaled, bar_width, label=index_list, color=colors)

    # Setting labels and title
    ax.set_title('Scaled single score of the Functional Unit')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(flow_legend)
    #plt.xticks(rotation=90)
    plt.yticks(np.arange(0, 1.01, step=0.1))
    plt.ylim(0,1.03)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'scaled_single_score_multi_{db_type}.png'), bbox_inches='tight')
    plt.show()

def process_categorizing(df_GWP, db_type, database_name, case, flow_legend, columns):
    x_axis = []
    GWP_value = []
    raw_dct = {}
    rec_dct = {}
    comp  = {}
    for col in df_GWP.columns:
        
        for i, row in df_GWP.iterrows():
            # print(f'idx = {i}')
            lst_x = []
            lst_GWP = []
            gwp_tot = 0
            for lst in row[col]:
                x = lst[0]
                gwp = lst[1]
                
                # if 'break even' in case.lower():
                #     x, gwp = break_even_flow_seperation(x, gwp, db_type, database_name)
                # else:   
                    
                x, gwp = flow_name_update(x, gwp, db_type, database_name)
                # # Updating the name of process
                # if 'Avoided mat. prod.' in x and gwp > 0:
                #     gwp = -gwp
                lst_x.append(x)
                lst_GWP.append(gwp)
                gwp_tot += gwp


            # print(gwp_tot, lst_GWP)
            lst_GWP.append(gwp_tot)
            lst_x.append('Net impact')
            x_axis.append(lst_x)
            GWP_value.append(lst_GWP)

    for key, item in raw_dct.items():
        comp[key] = rec_dct[key]/item*100

    # Create an empty dictionary to collect the data
    key_dic = {}

    # Collect the data into the dictionary
    for i, p in enumerate(GWP_value):
        for a, b in enumerate(p):
            key = (flow_legend[i], x_axis[i][a])
            # print(x_axis[i][a])

            if key in key_dic:
                key_dic[key] += b
            else:
                key_dic[key] = b
            # print(key)

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(list(key_dic.items()), columns=['Category', 'Value'])

    # Separate 'Total' values from the rest
    totals_df = df[df['Category'].apply(lambda x: x[1]) == 'Net impact']
    df = df[df['Category'].apply(lambda x: x[1]) != 'Net impact']

    # Pivot the DataFrame to have a stacked format
    df_stacked = df.pivot_table(index=[df['Category'].apply(lambda x: x[0])], columns=[df['Category'].apply(lambda x: x[1])], values='Value', aggfunc='sum').fillna(0)

    # Create a DataFrame to store results

    df_stack_updated = pd.DataFrame(0, index=flow_legend, columns=columns[:-1], dtype=object)  # dtype=object to handle lists
    for col in df_stack_updated.columns:
        for inx, row in df_stack_updated.iterrows():
            # print(df_stacked[col])
            try:
                row[col] = df_stacked[col][inx]

            except KeyError:
                # print(f"keyerror at {inx}")
                pass

                
    return df_stack_updated, totals_df

def gwp_lc_plot(df_GWP, category_mapping, categories, inputs, y_axis_values):
    import os
    
    flow_legend = inputs[0]
    colors = inputs[1]
    save_dir = inputs[2]
    db_type = inputs[3]
    database_name = inputs[4]

    x_axis, GWP_value, plot_legend = process_categorizing(df_GWP, db_type, database_name, category_mapping, '')

    color_map = {}
    #unique_processes = {process for sublist in x_axis for process in sublist}
    for i, process in enumerate(plot_legend):
        color_map[process] = colors[i]
        #print(process, i)


    # Initialize an ordered dictionary for legend_handles to maintain the order
    legend_handles = OrderedDict()

    # Initialize legend_handles with keys from plot_legend and empty lists
    for process in plot_legend:
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
        plt.savefig(os.path.join(save_dir, f'GWP_life_stage_{db_type}.png'), bbox_inches='tight')
        plt.show()
        

    else:
        print('The x-axis and GWP values have different sizes')

def category_organization(database_name):

    if 'case1' in database_name :
        category_mapping = {
        "Raw mat. + prod.": ["Raw mat. + prod."],
        "Use": ["Disinfection", "Autoclave"],
        "Recycling": ["Recycling"],
        "EoL": ["Incineration", "Avoided energy prod.", "Avoided mat. prod." ],
        "Net impact": ["Net impact"]
        }
    
    elif 'case2' in database_name or 'model' in database_name:
        category_mapping = {
        "Raw mat. + prod.": ["Raw mat. + prod."],
        "Use": ["Use",  "Disinfection",  "Ster. consumables", "Ster. autoclave"],
        "EoL": ["Incineration", "Avoided energy prod."],
        "Net impact": ["Net impact"]
        }

    return category_mapping        

# Function to plot the global warming potentials showing the contribution of each life stage
def scaled_FU_plot(df_scaled, plot_x_axis, inputs, impact_category, legend_position):
    # Configure general plot settings
    plt.rcParams.update({
    'font.size': 12,      # General font size
    'axes.titlesize': 14, # Title font size
    'axes.labelsize': 12, # Axis labels font size
    'legend.fontsize': 10 # Legend font size
    }) 
    flow_legend, colors, save_dir, db_type, database_name = inputs

    # Extract columns and indices for plotting
    columns_to_plot = df_scaled.columns
    index_list = list(df_scaled.index.values)

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_width = 1 / (len(index_list) + 1)
    index = np.arange(len(columns_to_plot))

    # Plot each group of bars
    for i, process in enumerate(df_scaled.index):
        values = df_scaled.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax.bar(index + i * bar_width, values, bar_width, label=process, color=color)

    # Format impact category string
    lcia_string = impact_category[0][0]
    lcia_string = lcia_string.replace(' Runes edition', '').replace('ReCiPe 2016 v1.03, ', '')

    # Format database type
    if 'cut' in db_type.lower():
        db_type = db_type.replace('_', '-').capitalize()
    else:
        db_type = db_type.upper()

    # Set title and labels
    ax.set_title(
        f'Scaled Impact of the Functional Unit - {lcia_string.capitalize()} ({db_type})',
         fontsize=14
    )

     
    ax.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax.set_xticklabels(plot_x_axis, fontsize=10)
    

    # Set x-axis limits based on case type
    if 'case1' in database_name:
        ax.set_xlim(-0.2, len(columns_to_plot))
    elif 'case2' in database_name:
        ax.set_xlim(-0.35, len(columns_to_plot) - 0.3)
    else:
        ax.set_xlim(0, len(columns_to_plot))

    # Customize tick sizes
    
    plt.yticks(fontsize=11)

    if 'endpoint' in lcia_string.lower():
        plt.xticks(rotation=0, fontsize=11)
        x_pos = -0.05
    else:
        plt.xticks(rotation=90, fontsize=11)
        x_pos = -0.17

    # Configure legend placement and formatting
    if len(df_scaled.index) <= 6:
        ax.legend(
            flow_legend,
            loc='upper center',
            bbox_to_anchor=(0.5, x_pos),
            ncol= len(df_scaled.index),  # Adjust the number of columns based on legend size
            fontsize=10,
            frameon=False
        )
    else:
        ax.legend(
            flow_legend,
            loc='upper center',
            bbox_to_anchor=(0.5, x_pos),
            ncol= len(df_scaled.index)/2,  # Adjust the number of columns based on legend size
            fontsize=10,
            frameon=False
        )

    # Save the plot with high resolution
    output_file = os.path.join(
        save_dir,
        f'scaled_impact_score_multi_{db_type}_{impact_category[0][0]}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def gwp_scenario_plot(df_GWP, inputs, y_axis_values):
    # Using the inputs to specify the different variables
    flow_legend = inputs[0]
    colors = inputs[1]
    save_dir = inputs[2]
    db_type = inputs[3]
    database_name = inputs[4]

    plt.rcParams.update({
    'font.size': 12,      # General font size
    'axes.titlesize': 14, # Title font size
    'axes.labelsize': 12, # Axis labels font size
    'legend.fontsize': 10 # Legend font size
    }) 

    columns = lc.unique_elements_list(database_name)
    df_stack_updated, totals_df = process_categorizing(df_GWP, db_type, database_name, '', flow_legend, columns)

    y_min, y_max, steps, leg_pos, marker_offset, marker_color = y_axis_values

    # Plotting the stacked bar chart
    fig, ax = plt.subplots(figsize=(7, 5))  # Adjusted for single-column width (~3.54 inches at 300 dpi)
    df_stack_updated.plot(kind='bar', stacked=True, ax=ax, color=colors, zorder=2)
    ax.axhline(y=0, color='k', linestyle='-', zorder=0, linewidth=0.5)

    # Plotting 'Total' values as dots and including it in the legend
    for idx, row in totals_df.iterrows():
        unit = row['Category'][0]
        total = row['Value']
        ax.plot(unit, total, 'D', color=marker_color, markersize=4, mec='k', label='Net impact' if idx == 0 else "")
        # Add the data value
        ax.text(
            unit, total - marker_offset, f"{total:.2f}", 
            ha='center', va='bottom', fontsize=10, 
            color=marker_color)

    # Custom legend with 'Total' included
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=marker_color, mec='k', markersize=4, label='Net impact')
    )

    ax.legend(
        labels=columns,
        handles=handles,
        loc="upper center",  # Place legend at the bottom center
        bbox_to_anchor=(0.5, -0.08),  # Adjust position to below the x-axis
        ncol=3,  # Display legend entries in 3 columns
        fontsize=10.5,
        frameon=False  # Remove the legend box
    )

    # Adjust database type for title
    if 'cut' in db_type.lower():
        db_type = db_type.replace('_', '-').capitalize()
    else:
        db_type = db_type.upper()

    # Setting labels and title
    plt.title(f'GWP Impact of Each Life Stage for 1 FU ({db_type})',  fontsize=14)
    plt.ylabel('Global Warming Potential [kg CO$_2$e/FU]',  fontsize=12)
    plt.yticks(np.arange(y_min, y_max + 0.01, step=steps), fontsize=11)
    plt.ylim(y_min - 0.05, y_max + 0.05)
    plt.xticks(rotation=0, fontsize=11)
    plt.tight_layout()

    # Save the figure
    filename = os.path.join(save_dir, f'GWP_life_stage_pr_scenario_{db_type}.png')
    plt.savefig(filename, dpi=300, format='png', bbox_inches='tight')  # Save with 300 dpi resolution
    plt.show()

def break_even_orginization(df_be, database_name):
    df_be_copy = dc(df_be)
    if 'case1' in database_name:
        wipe_small_container = df_be.at['ASW', 'Disinfection']
        wipe_large_container = df_be.at['ALW', 'Disinfection']

        # Avoided energy
        cabinet_small_avoided_energy = df_be.at['ASC', 'Avoided energy prod.']
        wipe_small_avoided_energy = df_be.at['ASW', 'Avoided energy prod.']

        allocate_avoided_energy_S = wipe_small_avoided_energy - cabinet_small_avoided_energy

        cabinet_large_avoided_energy = df_be.at['ALC', 'Avoided energy prod.']
        wipe_large_avoided_energy = df_be.at['ALW', 'Avoided energy prod.']

        allocate_avoided_energy_L = wipe_large_avoided_energy - cabinet_large_avoided_energy



        # Incineration
        cabinet_small_inc = df_be.at['ASC', 'Incineration']
        wipe_small_inc = df_be.at['ASW', 'Incineration']

        allocate_inc_S = wipe_small_inc - cabinet_small_inc

        cabinet_large_inc = df_be.at['ALC', 'Incineration']
        wipe_large_inc = df_be.at['ALW', 'Incineration']

        allocate_inc_L = wipe_large_inc - cabinet_large_inc

        # Recycling
        cabinet_small_rec = df_be.at['ASC', 'Recycling']
        wipe_small_rec = df_be.at['ASW', 'Recycling']

        allocate_rec_S = wipe_small_rec - cabinet_small_rec

        cabinet_large_rec = df_be.at['ALC', 'Recycling']
        wipe_large_rec = df_be.at['ALW', 'Recycling']

        allocate_rec_L = wipe_large_rec - cabinet_large_rec

        # Calculating the new sums

        wipe_small_container_new = wipe_small_container + allocate_avoided_energy_S + allocate_inc_S + allocate_rec_S

        wipe_large_container_new = wipe_large_container + allocate_avoided_energy_L  + allocate_inc_L +allocate_rec_L


        df_be_copy.at['ASW', 'Avoided energy prod.'] = cabinet_small_avoided_energy
        df_be_copy.at['ALW', 'Avoided energy prod.'] = cabinet_large_avoided_energy

        df_be_copy.at['ASW', 'Incineration'] = cabinet_small_inc
        df_be_copy.at['ALW', 'Incineration'] = cabinet_large_inc

        df_be_copy.at['ASW', 'Disinfection'] = wipe_small_container_new
        df_be_copy.at['ALW', 'Disinfection'] = wipe_large_container_new

        df_be_copy.at['ASW', 'Recycling'] = cabinet_small_rec
        df_be_copy.at['ALW', 'Recycling'] = cabinet_large_rec

    return df_be_copy

def break_even_graph(df_GWP, inputs, plot_structure):
    # Unpack inputs
    flow_legend, colors, save_dir, db_type, database_name = inputs
    amount_of_uses, y_max, ystep, xstep, break_even_product, color_idx = plot_structure

    columns = lc.unique_elements_list(database_name)
    case = 'break even'
    df_be, ignore = process_categorizing(df_GWP, db_type, database_name, case, flow_legend, columns)

    plt.rcParams.update({
            'font.size': 12,      # General font size
            'axes.titlesize': 14, # Title font size
            'axes.labelsize': 12, # Axis labels font size
            'legend.fontsize': 10 # Legend font size
            }) 

    if 'case1' in database_name:
        df_be_copy = break_even_orginization(df_be, database_name)
        # Split index into small and large based on criteria
        small_idx = [idx for idx in df_be_copy.index if '2' in idx or 'AS' in idx]
        large_idx = [idx for idx in df_be_copy.index if idx not in small_idx]

        # Create empty DataFrames for each scenario
        scenarios = {
            'small': pd.DataFrame(0, index=small_idx, columns=df_be_copy.columns, dtype=object),
            'large': pd.DataFrame(0, index=large_idx, columns=df_be_copy.columns, dtype=object)
        }

        # Fill scenarios with data
        for sc_idx, (scenario_name, scenario_df) in enumerate(scenarios.items()):
            scenario_df.update(df_be_copy.loc[scenario_df.index])

            use_cycle, production = {}, {}

            for idx, row in scenario_df.iterrows(): 
                use, prod = 0, 0
                for col in df_be_copy.columns:
                    if ('autoclave' in col.lower() or 'disinfection' in col.lower()) and 'H' not in idx:
                        use_cycle[idx] = row[col] + use
                        use += row[col]
                    elif 'A' in idx:
                        # print(idx, col ,(row[col] + prod) * amount_of_uses)
                        production[idx] = (row[col] + prod) * amount_of_uses
                        prod += row[col]
                        
                    else:
                        production[idx] = row[col] + prod
                        prod += row[col]
            
            # Calculate break-even values
            be_dct = {}
            for key, usage in production.items():
                be_dct[key] = [usage if u == 1 else use_cycle.get(key, usage) * u + usage
                            for u in range(1, amount_of_uses + 1)]

            # Plot results
            _, ax = plt.subplots(figsize=(7, 5))
            
            

            for idx, (key, value) in enumerate(be_dct.items()):
                try:
                    if 'H' in key:
                        ax.plot(value, label=key,linestyle='dashed', color=colors[color_idx[idx] % len(colors)], linewidth=3)
                    else:
                        ax.plot(value, label=key, color=colors[color_idx[idx]], linewidth=3)
                except IndexError:
                    print(f'Color index of {color_idx[idx]} is out of range, choose a value between 0 and {len(colors) - 1}')        
            
            if 'cut' in db_type.lower():
                db_type = db_type.replace('_', '-')
                db_type = db_type.capitalize()
            else:
                db_type= db_type.upper()

            # Customize plot
            ax.legend(
                loc="upper center",  # Place legend at the bottom center
                bbox_to_anchor=(0.5, -0.15),  # Adjust position to below the x-axis
                ncol=4,  # Display legend entries in 3 columns
                fontsize=10.5,
                frameon=False  # Remove the legend box
            )
            plt.title(f'Break even for the {scenario_name} {break_even_product} ({db_type})')
            plt.xlabel('Cycle(s)',  )
            plt.ylabel('Accumulated Global Warming Pot. [kg CO$_2$e]')
            plt.xlim(0, amount_of_uses)
            plt.xticks(range(0, amount_of_uses, xstep))

            plt.ylim(0, y_max[sc_idx]+ 1)
            plt.yticks(range(0, y_max[sc_idx] + 1, ystep[sc_idx]))
            plt.tight_layout()

            # Save and display plot
            filename = os.path.join(save_dir, f'break_even_{scenario_name}_{db_type}.png')
            plt.savefig(filename, dpi=300, format='png', bbox_inches='tight')  # Save with 300 dpi resolution
            plt.show()

        
    elif 'case2' in database_name:
        multi_use, production = {}, {}

        for idx, row in df_be.iterrows(): 
            use, prod = 0, 0
            for col in df_be.columns:
                if 'Disinfection' in col and 'SUD' not in idx:
                    multi_use[idx] = row[col] + use
                    use += row[col]
                elif 'MUD' in idx:
                    production[idx] = (row[col] + prod) * amount_of_uses
                    prod += row[col]
                else:
                    production[idx] = row[col] + prod
                    prod += row[col]

        # Calculate break-even values
        be_dct = {}
        for key, usage in production.items():
            be_dct[key] = [usage if u == 1 else multi_use.get(key, usage) * u + usage
                        for u in range(1, amount_of_uses + 1)]

        # Plot results
        fig, ax = plt.subplots(figsize=(7, 5))

        

        for idx, (key, value) in enumerate(be_dct.items()):
            # if color_idx == 0:
            try:
                if 'RMD' in key or 'SUD' in key:
                    ax.plot(value, label=key,linestyle='dashed', color=colors[color_idx[idx] % len(colors)], linewidth=3)
                else:
                    ax.plot(value, label=key, color=colors[color_idx[idx]], linewidth=3)
            except IndexError:
                print(f'Color index of {color_idx[idx]} is out of range, choose a value between 0 and {len(colors) - 1}')        
        if 'cut' in db_type.lower():
            db_type = db_type.replace('_', '-')
            db_type = db_type.capitalize()
        else:
            db_type= db_type.upper()

        # Customize plot
        ax.legend(
                loc="upper center",  # Place legend at the bottom center
                bbox_to_anchor=(0.5, -0.15),  # Adjust position to below the x-axis
                ncol=4,  # Display legend entries in 3 columns
                fontsize=10.5,
                frameon=False  # Remove the legend box
            )
        plt.title(f'Break even for the {break_even_product} ({db_type})', )
        plt.xlabel('Cycle(s)',  )
        plt.ylabel('Accumulated Global Warming Pot. [kg CO$_2$e]',  )
        plt.xlim(0, amount_of_uses)
        plt.xticks(range(0, amount_of_uses +1, xstep))

        plt.ylim(0, y_max[0]+ 10)
        plt.yticks(range(0, y_max[0] + 1, ystep[0]))
        plt.tight_layout()


        # Save and display plot
        filename = os.path.join(save_dir, f'break_even_bipolar_{db_type}.png')
        plt.savefig(filename, dpi=300, format='png', bbox_inches='tight')  # Save with 300 dpi resolution
        plt.show()
   

def create_results_graphs(initialization, df, plot_x_axis_all, save_dir, impact_categories, flow_legend, plot_structure):
    leg_pos_scal, leg_pos_gwp_, y_min_gwp, y_max_gwp, y_step_gwp, amount_of_uses, y_max_be, y_step_be, x_step_be_, be_product = plot_structure
    
    leg_pos_scaled = {key: pos for case, pos in leg_pos_scal.items() for key in initialization.keys() if case in key}
    leg_pos_gwp = {key: pos for case, pos in leg_pos_gwp_.items() for key in initialization.keys() if case in key}
    life_time_use = {key: use for case, use in amount_of_uses.items() for key in initialization.keys() if case in key}
    x_step_be = {key: x_val for case, x_val in x_step_be_.items() for key in initialization.keys() if case in key}
    break_even_product = {key: be_name for case, be_name in be_product.items() for key in initialization.keys() if case in key}

    for key, item in initialization.items():
        database_name = item[1]

        df_res, plot_x_axis_lst = lc.dataframe_results_handling(df[key], database_name, plot_x_axis_all[key], item[3])
        if type(df_res) is list:
            df_mid, df_endpoint = df_res
            plot_x_axis, plot_x_axis_end = plot_x_axis_lst


        _, df_scaled = lc.dataframe_element_scaling(df_mid)
        df_col = [df_mid.columns[1]]
        df_GWP = df_mid[df_col]

        cmap = plt.get_cmap('Accent')
        colors = [cmap(i) for i in np.linspace(0, 1, 9)]
        inputs = [flow_legend[key], colors, save_dir[key], item[4], database_name]

        scaled_FU_plot(df_scaled, plot_x_axis, inputs, impact_categories[key], leg_pos_scaled[key])

        if 'recipe' in item[3].lower():
            _, df_scaled_e = lc.dataframe_element_scaling(df_endpoint)
            scaled_FU_plot(df_scaled_e, plot_x_axis_end, inputs, impact_categories[key][-3:], leg_pos_scaled[key])

        marker_offset = 0.15
        marker_color = 'k'
        y_axis_values = [y_min_gwp[key], y_max_gwp[key], y_step_gwp[key], leg_pos_gwp[key], marker_offset, marker_color]
        gwp_scenario_plot(df_GWP, inputs, y_axis_values)

        color_idx = [0, 1, 2, 4]

        plot_controls = [life_time_use[key], y_max_be[key], y_step_be[key], x_step_be[key], break_even_product[key], color_idx]
        break_even_graph(df_GWP, inputs, plot_controls)