import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from matplotlib.lines import Line2D  # Import for creating custom legend markers
import importlib
import os

from  standards import *
import life_cycle_assessment as lc
importlib.reload(lc)

# Function to update the flow name and simplify them
def flow_name_update(x, gwp, db_type, database_name):
    if 'Ananas consq' in database_name or 'sterilization' in database_name:
        if f'- {db_type}' in x:
            #print(key)
            x = x.replace(f' - {db_type}', '')
        x_og = x
        if 'alubox' in x:       

            x = x.replace('alubox ', '')
            if 'avoided' in x:
                x = 'Avoided mat. prod.'
                if gwp < 0:
                    gwp = -gwp
            if 'raw materials' in x:
                x = 'Raw mat.' 
                if gwp < 0:
                    gwp = -gwp 
            if 'production' in x:
                x = 'Manufacturing' 
            if 'EoL' in x:
                x = 'Recycling'
        if 'waste paper to pulp' in x:
            x = 'Avoided mat. prod.'
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
        if 'no Energy Recovery' in x or 'incineration' in x:
            x = 'Incineration'

        if 'board box' in x or 'packaging film' in x:
            x = 'Packaging'

        if 'autoclave' in x:
            x = 'Autoclave'
        if 'transport' in x:
            x = 'Transport'
        if 'cabinet' in x or 'wipe' in x:
            x = 'Container cleaning'
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

    return x, gwp

# Function to create the scaled FU plot
def scaled_FU_plot(df_scaled, plot_x_axis, inputs, impact_category, legend_position):
    
    flow_legend = inputs[0]
    colors = inputs[1]
    save_dir = inputs[2]
    db_type = inputs[3]

    # Extracting the columns plot
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

    # Setting labels and title
    ax.set_title(f'Scaled impact of the Functional Unit - {impact_category[0][0]} {db_type}',weight='bold',fontsize=16)
    ax.set_xticks(index + bar_width * (len(index_list) - 1) / 2)
    ax.set_xticklabels(plot_x_axis,  weight='bold', fontsize=12)

    # Specifying the direction of the text on the axis should be rotated
    if 'endpoint' not in impact_category[0][0]:
        plt.xticks(rotation=90)
    else: 
        plt.xticks(rotation=0)

    # Setting the legend
    ax.legend(flow_legend,bbox_to_anchor=(1.01, legend_position, .1, 0), loc="lower left",
                mode="expand", borderaxespad=0,  ncol=1, fontsize=10)
    
    # Saving and showing the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'scaled_impact_score_multi_{db_type}_{impact_category[0][0]}.jpg'), bbox_inches='tight')
    plt.show()

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
        "Use": ["Autoclave", "Container cleaning"],
        "Transport": ["Transport"],
        "EoL": ["Incineration", "Recycling", "Avoided mat. prod.", "Avoided energy prod."],
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

# Function to plot the global warming potentials showing the contribution of each life stage
def gwp_scenario_plot(df_GWP, inputs, y_axis_values):
    # Using the inputs to specify the different variables
    flow_legend = inputs[0]
    colors = inputs[1]
    save_dir = inputs[2]
    db_type = inputs[3]
    database_name = inputs[4]

    columns = lc.unique_elements_list(database_name)


    x_axis = []
    GWP_value = []

    # Obtaining the contribution of each scenario for the life stages 
    for col in df_GWP.columns:
        for i, row in df_GWP.iterrows():
            # Empty list to store the process name and the value
            lst_x = []
            lst_GWP = []
            # initial value for the total value of the gwp total
            gwp_tot = 0
            # Iterating over each nested list
            for lst in row[col]:
                x = lst[0]
                gwp = lst[1]

                # Updating the name of process
                x, gwp = flow_name_update(x, gwp, db_type, database_name)
                if 'Avoided mat. prod.' in x and gwp > 0:
                    gwp = -gwp

                lst_x.append(x)
                lst_GWP.append(gwp)
                gwp_tot += gwp
            
            # Setting the updated list back into a new nested list
            lst_GWP.append(gwp_tot)
            lst_x.append('Total')
            x_axis.append(lst_x)
            GWP_value.append(lst_GWP)

    # Create an empty dictionary to collect the data
    key_dic = {}

    # Collect the data into the dictionary
    for i, p in enumerate(GWP_value):
        for a, b in enumerate(p):
            key = (flow_legend[i], x_axis[i][a])
            if key in key_dic:
                key_dic[key] += b
            else:
                key_dic[key] = b

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(list(key_dic.items()), columns=['Category', 'Value'])

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

    y_min = y_axis_values[0]
    y_max = y_axis_values[1]
    steps = y_axis_values[2]
    leg_pos = y_axis_values[3]
    marker_offset = y_axis_values[4]

    # Plotting the stacked bar chart
    ax = df_stack_updated.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)
    ax.axhline(y = -0.004, color = 'k', linestyle = '-', zorder=0, linewidth=0.5) # https://matplotlib.org/stable/gallery/misc/zorder_demo.html
     
    # Plotting 'Total' values as dots and including it in the legend
    for flow in flow_legend:
        for idx, row in totals_df.iterrows():
            if flow in row['Category'][0]:
                unit = row['Category'][0]
                total = row['Value']
                ax.plot(unit, total, 'D', color='k', markersize=5, label='Total' if idx == 0 else "")
                # Add the data value
                ax.text(unit, total - marker_offset, f"{total:.2f}", ha='center', va='bottom', fontsize=9) # https://www.datacamp.com/tutorial/python-round-to-two-decimal-places?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720824&utm_adgroupid=157156376311&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=684592140434&utm_targetid=dsa-2218886984100&utm_loc_interest_ms=&utm_loc_physical_ms=9197406&utm_content=&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-row-p2_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na-oct24&gad_source=1&gclid=Cj0KCQiA_qG5BhDTARIsAA0UHSK7fmd8scMcHSkG_VMO1TWmeHapAM6cjV1QobZKKYotZPX7IcmJRF4aAhsyEALw_wcB

    

    # Custom legend with 'Total' included
    handles, labels = ax.get_legend_handles_labels()

    handles.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='k', markersize=6, label='Total'))
    ax.legend(labels=columns, handles=handles, bbox_to_anchor=(1.01, leg_pos, .23, 0), loc="lower left", mode="expand", borderaxespad=0, ncol=1, fontsize=10)



    # Setting labels and title
    plt.title(f'GWP impact for each life stage for 1 FU - {db_type}', weight='bold')
    plt.ylabel('Global Warming Potential [kg CO$_2$e]', weight='bold')
    plt.yticks(np.arange(y_min, y_max + 0.01, step=steps))
    plt.ylim(y_min-0.05, y_max+0.05)
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'GWP_life_stage_pr_scenario_{db_type}.jpg'), bbox_inches='tight')
    plt.show()

    return df_stack_updated

def break_even_graph(df_stacked, inputs, plot_structure):
    # Unpack inputs
    colors, save_dir, db_type = inputs[1], inputs[2], inputs[3]
    amount_of_uses, y_max, ystep, xstep, break_even_product, color_idx = plot_structure

    # Split index into small and large based on criteria
    small_idx = [idx for idx in df_stacked.index if '2' in idx or 'AS' in idx]
    large_idx = [idx for idx in df_stacked.index if idx not in small_idx]

    # Create empty DataFrames for each scenario
    scenarios = {
        'small': pd.DataFrame(0, index=small_idx, columns=df_stacked.columns, dtype=object),
        'large': pd.DataFrame(0, index=large_idx, columns=df_stacked.columns, dtype=object)
    }

    # Fill scenarios with data
    for sc_idx, (scenario_name, scenario_df) in enumerate(scenarios.items()):
        scenario_df.update(df_stacked.loc[scenario_df.index])

        alu_box_use, production = {}, {}

        for idx, row in scenario_df.iterrows(): 
            use, prod = 0, 0
            for col in df_stacked.columns:
                if ('Autoclave' in col or 'Box cleaning' in col) and 'H' not in idx:
                    alu_box_use[idx] = row[col] + use
                    use += row[col]
                elif 'A' in idx:
                    production[idx] = (row[col] + prod) * amount_of_uses
                    prod += row[col]
                else:
                    production[idx] = row[col] + prod
                    prod += row[col]

        # Calculate break-even values
        be_dct = {}
        for key, usage in production.items():
            be_dct[key] = [usage if u == 1 else alu_box_use.get(key, usage) * u + usage
                           for u in range(1, amount_of_uses + 1)]

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))

        

        for idx, (key, value) in enumerate(be_dct.items()):
            # if color_idx == 0:
            try:
                if 'H' in key:
                    ax.plot(value, label=key,linestyle='dashed', color=colors[color_idx[idx] % len(colors)], markersize=3.5)
                else:
                    ax.plot(value, label=key, color=colors[color_idx[idx]], markersize=3.5)
            except IndexError:
                print(f'Color index of {color_idx[idx]} is out of range, choose a value between 0 and {len(colors) - 1}')
            # else:
            #     if 'H' in key:
            #         ax.plot(value, label=key,linestyle='dashed', color=colors[color_idx + 2], markersize=3.5)
            #     else:
            #         ax.plot(value, label=key, color=colors[color_idx + 2], markersize=3.5)
            
        #ax.axhline(y = 0, color = 'k', linestyle = '-', zorder=0, linewidth=0.5) # https://matplotlib.org/stable/gallery/misc/zorder_demo.html

        

        # Customize plot
        ax.legend(bbox_to_anchor=(1.00, 1.017), loc='upper left')
        plt.title(f'Break even for the {scenario_name} {break_even_product} - {db_type}', weight = 'bold')
        plt.xlabel('Cleaning operation(s)',  weight = 'bold')
        plt.ylabel('Global Warming Potential [kg CO$_2$e]',  weight = 'bold')
        plt.xlim(0, amount_of_uses)
        plt.xticks(range(0, amount_of_uses, xstep))

        plt.ylim(0, y_max[sc_idx])
        plt.yticks(range(0, y_max[sc_idx] + 20, ystep[sc_idx]))
        plt.tight_layout()

        # Save and display plot
        plt.savefig(os.path.join(save_dir, f'break_even_{scenario_name}_{db_type}.jpg'), bbox_inches='tight')
        plt.show()