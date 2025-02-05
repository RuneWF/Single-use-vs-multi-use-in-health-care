import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from matplotlib.lines import Line2D  # Import for creating custom legend markers
import os
from copy import deepcopy as dc

from  standards import *
import life_cycle_assessment as lc

def plot_title_text(lca_type):
    if 'consq' in lca_type:
        return 'Consequential'
    elif 'cut' in lca_type:
        return 'Allocation cut-off by Classification'
    else:
        return ''

def join_path(path1, path2):
    return os.path.join(path1, path2)

# Function to update the flow name and simplify them
def flow_name_update(x, gwp, database_name):
    x_og = x


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
        if '18/8' in x:
            x = "Avoided mat. prod."

    elif 'case2' in database_name or 'model' in database_name:
        if 'H200' in x:
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

def process_categorizing(df_GWP, database_name, flow_legend, columns):
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
                    
                x, gwp = flow_name_update(x, gwp, database_name)

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

    # Set title and labels
    ax.set_title(plot_title_text(db_type), fontsize=14)

     
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
    df_stack_updated, totals_df = process_categorizing(df_GWP, database_name, flow_legend, columns)

    y_min, y_max, steps, _, marker_offset, marker_color = y_axis_values

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


    # Setting labels and title
    ax.set_title(plot_title_text(db_type), fontsize=14)
    plt.ylabel('Global Warming Potential [kg CO$_2$e/FU]',  fontsize=12)
    plt.yticks(np.arange(y_min, y_max + 0.01, step=steps), fontsize=11)
    plt.ylim(y_min - 0.05, y_max + 0.05)
    plt.xticks(rotation=0, fontsize=11)
    plt.tight_layout()

    # Save the figure
    filename = join_path(save_dir, f'GWP_life_stage_pr_scenario_{db_type}.png')
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
    amount_of_uses, y_max, ystep, xstep, _, color_idx = plot_structure

    columns = lc.unique_elements_list(database_name)
    df_be, ignore = process_categorizing(df_GWP, database_name, flow_legend, columns)

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

            # Customize plot
            ax.legend(
                loc="upper center",  # Place legend at the bottom center
                bbox_to_anchor=(0.5, -0.15),  # Adjust position to below the x-axis
                ncol=4,  # Display legend entries in 3 columns
                fontsize=10.5,
                frameon=False  # Remove the legend box
            )
            ax.set_title(f"{scenario_name.capitalize()} - {plot_title_text(db_type)}", fontsize=14)
            plt.xlabel('Cycle(s)',  )
            plt.ylabel('Accumulated Global Warming Pot. [kg CO$_2$e]')
            plt.xlim(0, amount_of_uses)
            plt.xticks(range(0, amount_of_uses, xstep))

            plt.ylim(0, y_max[sc_idx]+ 1)
            plt.yticks(range(0, y_max[sc_idx] + 1, ystep[sc_idx]))
            plt.tight_layout()

            # Save and display plot
            filename = join_path(save_dir, f'break_even_{scenario_name}_{db_type}.png')
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

        # Customize plot
        ax.legend(
                loc="upper center",  # Place legend at the bottom center
                bbox_to_anchor=(0.5, -0.15),  # Adjust position to below the x-axis
                ncol=4,  # Display legend entries in 3 columns
                fontsize=10.5,
                frameon=False  # Remove the legend box
            )
        ax.set_title(f"{plot_title_text(db_type)}", fontsize=14)
        plt.xlabel('Cycle(s)',  )
        plt.ylabel('Accumulated Global Warming Pot. [kg CO$_2$e]',  )
        plt.xlim(0, amount_of_uses)
        plt.xticks(range(0, amount_of_uses +1, xstep))

        plt.ylim(0, y_max[0]+ 10)
        plt.yticks(range(0, y_max[0] + 1, ystep[0]))
        plt.tight_layout()


        # Save and display plot
        filename = join_path(save_dir, f'break_even_bipolar_{db_type}.png')
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
        print(f"Creating figures for {key}")
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

def color_range():
    cmap = plt.get_cmap('Accent')
    return [cmap(i) for i in np.linspace(0, 1, 9)]

def legend_text(text):
    if '1' in text:
        flow_leg = [
                        'H2I',
                        'H2R',
                        'ASC',
                        'ASW',
                        'H4I',
                        'H4R',
                        'ALC',
                        'ALW'
                        ]
        return flow_leg
    else:
        return ['SUD', 'MUD']

def dataframe_for_plots(initialization, flow_legend, plot_x_axis_all):
    data = {
    'case1' : {},
    'case2' : {}
    }

    for key, item in initialization.items():

        database_name = item[1]
        if 'apos' not in database_name:
            if '1' in database_name:
                df_res, plot_x_axis_lst = lc.dataframe_results_handling(df[key], database_name, plot_x_axis_all[key], item[3])
                if type(df_res) is list:
                    df_mid, df_endpoint = df_res
                    plot_x_axis, plot_x_axis_end = plot_x_axis_lst


                _, df_scaled = lc.dataframe_element_scaling(df_mid)
                df_col = [df_mid.columns[1]]
                df_GWP = df_mid[df_col]

                if 'recipe' in item[3].lower():
                    _, df_scaled_e = lc.dataframe_element_scaling(df_endpoint)
                
                # inputs = [flow_legend[key], colors, save_dir[key], item[4], database_name]
                columns = lc.unique_elements_list(database_name)
                df_be, ignore = lpc.process_categorizing(df_GWP, database_name, flow_legend[key], columns)

                data['case1'].update({key : [df_scaled, df_scaled_e, df_GWP, df_be]})
            elif '2' in database_name:
                df_res, plot_x_axis_lst = lc.dataframe_results_handling(df[key], database_name, plot_x_axis_all[key], item[3])
                if type(df_res) is list:
                    df_mid, df_endpoint = df_res
                    plot_x_axis, plot_x_axis_end = plot_x_axis_lst


                _, df_scaled = lc.dataframe_element_scaling(df_mid)
                df_col = [df_mid.columns[1]]
                df_GWP = df_mid[df_col]

                if 'recipe' in item[3].lower():
                    _, df_scaled_e = lc.dataframe_element_scaling(df_endpoint)
                
                columns = lc.unique_elements_list(database_name)
                df_be, ignore = process_categorizing(df_GWP, database_name, flow_legend[key], columns)

                data['case2'].update({key : [df_scaled, df_scaled_e, df_GWP, df_be]})
    
    return data

def results_folder(path, case):
    folder = results_folder(join_path(path,'results'), case)
    return folder

def plot_font_sizes():
    plt.rcParams.update({
    'font.size': 12,      # General font size
    'axes.titlesize': 14, # Title font size
    'axes.labelsize': 12, # Axis labels font size
    'legend.fontsize': 10 # Legend font size
    }) 

def midpoint_plot(data, case, plot_x_axis, initialization, path):
    recipe = 'midpoint (H)'
    plot_font_sizes()
    df1 = data[case][f'{case}_cut_off'][0]
    df2 = data[case][f'{case}_consq'][0]
    colors = color_range()
    #  Extract columns and indices for plotting
    columns_to_plot = df1.columns
    index_list = list(df1.index.values)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9))
    bar_width = 1 / (len(index_list) + 1)
    index = np.arange(len(columns_to_plot))

    # Plot each group of bars
    for i, process in enumerate(df1.index):
        values = df1.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax1.bar(index + i * bar_width, values, bar_width, label=process, color=color)

    for i, process in enumerate(df2.index):
        values = df2.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax2.bar(index + i * bar_width, values, bar_width, label=process, color=color)

    # Format impact category string

    # Set title and labels
    ax1.set_title(f"{plot_title_text('cut')} - {recipe}")  
    ax1.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax1.set_xticklabels(plot_x_axis, rotation=90)  # Added rotation here
    ax1.set_xlim(-0.2, len(columns_to_plot))

    ax2.set_title(f"{plot_title_text('consq')} - {recipe}")
    ax2.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax2.set_xticklabels(plot_x_axis, rotation=90)  # Added rotation here
    ax2.set_xlim(-0.2, len(columns_to_plot))

    x_pos = 0.97


    fig.legend(
        legend_text(initialization[f'{case}_consq'][1]),
        loc='upper left',
        bbox_to_anchor=(0.965, x_pos),
        ncol= 1,  # Adjust the number of columns based on legend size
        fontsize=10,
        frameon=False
    )

    # Save the plot with high resolution
    output_file = join_path(
        results_folder(path, case),
        f'{recipe}_{case}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.show()

def end_plot(data, case, plot_x_axis_end, initialization, path):
    recipe = 'endpoint (H)'
    plot_font_sizes()
    df1 = data[case][f'{case}_cut_off'][0]
    df2 = data[case][f'{case}_consq'][0]
    colors = color_range()

    # Extract columns and indices for plotting
    columns_to_plot = df1.columns
    index_list = list(df1.index.values)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9))
    bar_width = 1 / (len(index_list) + 1)
    index = np.arange(len(columns_to_plot))

    # Plot each group of bars
    for i, process in enumerate(df1.index):
        values = df1.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax1.bar(index + i * bar_width, values, bar_width, label=process, color=color)

    for i, process in enumerate(df2.index):
        values = df2.loc[process, columns_to_plot].values
        color = colors[i % len(colors)]  # Ensure color cycling
        ax2.bar(index + i * bar_width, values, bar_width, label=process, color=color)

    # Format impact category string

    # Set title and labels
    ax1.set_title(f"{plot_title_text('cut')} - {recipe}")  
    ax1.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax1.set_xticklabels(plot_x_axis_end, rotation=0)  # Added rotation here
    ax1.set_xlim(-0.2, len(columns_to_plot))

    ax2.set_title(f"{plot_title_text('consq')} - {recipe}")
    ax2.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax2.set_xticklabels(plot_x_axis_end, rotation=0)  # Added rotation here
    ax2.set_xlim(-0.2, len(columns_to_plot))

    x_pos = 0.97


    fig.legend(
        legend_text(initialization[f'{case}_consq'][1]),
        loc='upper left',
        bbox_to_anchor=(0.965, x_pos),
        ncol= 1,  # Adjust the number of columns based on legend size
        fontsize=10,
        frameon=False
    )

    # Save the plot with high resolution
    output_file = os.path.join(
        results_folder(path, case),
        f'{recipe}_{case}.png'
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.show()