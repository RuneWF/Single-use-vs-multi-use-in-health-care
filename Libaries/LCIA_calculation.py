import multiprocessing as mp
import pandas as pd
import brightway2 as bw
import importlib
import os

import life_cycle_assessment as lc
importlib.reload(lc)


def lca_worker(args):
    proc, factor, impact = args
    try:
        if factor is None:
            print(f"Skipping {proc} as the factor is None.")
            return proc, impact, None

        lca = bw.LCA({proc: factor}, impact)
        lca.lci()
        lca.lcia()
        return proc, impact, lca.score
    except Exception as e:
        print(f"Error with {proc}, {impact}: {e}")
        return proc, impact, None



# Updated quick_LCIA_calculator with multiprocessing
def quick_LCIA_calculator(unique_process_index, uniquie_process_dct, impact_categories, file_name_unique, sheet_name, num_cores=4):
    df_unique = pd.DataFrame(0, index=unique_process_index, columns=impact_categories, dtype=object)
    unique_process_results = {}

    # Prepare tasks for multiprocessing using list comprehension
    tasks = [
        (proc, df_unique.at[proc, impact], impact)  # Exactly 3 elements: proc, factor, impact
        for proc in unique_process_index
        for impact in impact_categories
    ]

    # Remove the redundant loop to avoid duplicating tasks
    # The tasks are already created above in the list comprehension
    print(f"Prepared {len(tasks)} tasks.")  # Print number of tasks for debugging

    print(f"Starting multiprocessing with {num_cores} cores")

    # Use multiprocessing Pool
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(lca_worker, tasks)

    # Process results
    calc_count = 1
    total_calculations = len(results)
    for key, impact, score in results:
        if score is not None:
            # Assign the LCA score to the DataFrame
            col_idx = impact_categories.index(impact)
            row_idx = unique_process_index.index(str(key))
            df_unique.iat[row_idx, col_idx] = score

            # Print progress
            print(f"Calculation {calc_count}/{total_calculations}: {key}, Score: {score}")
        else:
            print(f"Failed calculation for {key} and impact {impact[1]}")

        calc_count += 1

    # Save the results
    lc.save_LCIA_results(df_unique, file_name_unique, sheet_name, impact_categories)

def redo_LCIA_unique_process(df_unique, initialization, unique_process_index, file_name_unique, sheet_name, num_cores=4):
    database_project, database_name, flows, lcia_method, db_type = initialization
    functional_unit, process, impact_category, plot_x_axis_all = lc.LCA_initialization(database_project, database_name, flows, lcia_method, db_type)

    # Ensure impact categories is a list
    impact_categories = list(impact_category) if isinstance(impact_category, tuple) else impact_category

    # Prepare tasks for multiprocessing
    tasks = []
    for proc in unique_process_index:
        if proc in df_unique.index:  # only include processes that are in the DataFrame
            tasks.extend(
                [(proc, df_unique.at[proc, impact], impact) for impact in impact_categories]
            )

    # Multiprocessing setup
    print(f"Starting multiprocessing with {num_cores} cores")
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(lca_worker, tasks)

    # Process results
    df_unique_copy = df_unique.copy()
    for proc, impact, score in results:
        if score is not None:
            df_unique_copy.at[proc, impact] = score

    # Save the results
    lc.save_LCIA_results(df_unique_copy, file_name_unique, sheet_name, impact_categories)
    return df_unique_copy

def quick_LCIA(initialization, file_name, file_name_unique, sheet_name, num_cores=4):
    database_project, database_name, flows, lcia_method, db_type = initialization
    functional_unit, process, impact_category, plot_x_axis_all = lc.LCA_initialization(database_project, database_name, flows, lcia_method, db_type)

    # Ensure impact categories is a list
    impact_categories = list(impact_category) if isinstance(impact_category, tuple) else impact_category

    # Prepare unique processes and indices
    unique_process_index = []
    uniquie_process_dct = {}

    for func_dict in functional_unit:
        for FU_key, FU_item in func_dict.items():
            for proc in FU_item.keys():
                if f'{proc}' not in unique_process_index:
                    unique_process_index.append(f'{proc}')
                    uniquie_process_dct[proc] = 1

    unique_process_index.sort()

    # Check if file exists
    if os.path.isfile(file_name_unique):
        try:
            df_unique = lc.import_LCIA_results(file_name_unique, unique_process_index, impact_category)
            user_input = input("Do you want to redo the calculations for some process (select 'a' to redo everything, or 'r' to recalculate based only on the FU)? [y/n]")

            if 'y' in user_input.lower():
                df_unique_new = redo_LCIA_unique_process(df_unique, initialization, unique_process_index, file_name_unique, sheet_name, num_cores)
            elif 'a' in user_input.lower():
                quick_LCIA_calculator(unique_process_index, uniquie_process_dct, impact_categories, file_name_unique, sheet_name, num_cores)

        except ValueError:
            print("ValueError encountered")
            quick_LCIA_calculator(unique_process_index, uniquie_process_dct, impact_categories, file_name_unique, sheet_name, num_cores)
    else:
        quick_LCIA_calculator(unique_process_index, uniquie_process_dct, impact_categories, file_name_unique, sheet_name, num_cores)

    if 'a' in user_input.lower() or 'y' in user_input.lower() or 'r' in user_input.lower():
        df_unique_new = lc.import_LCIA_results(file_name_unique, unique_process_index, impact_category)

        df = pd.DataFrame(0, index=flows, columns=impact_categories, dtype=object)
        for col in impact_categories:
            for i, row in df.iterrows():
                row[col] = []

        # Perform impact value calculations using multiprocessing
        tasks = []
        for col, impact in enumerate(impact_category):
            for fu in functional_unit:
                for key, item in fu.items():
                    proc = str([p for p in item.keys()][0])
                    val = float([v for v in item.values()][0])
                    factor = df_unique_new.at[proc, impact]
                    tasks.append((key, item, impact, val, factor))

        # Use multiprocessing for impact value calculations
        print(f"Starting multiprocessing with {num_cores} cores")
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(lca_worker, tasks)

        # Process results and assign to DataFrame
        for key, impact, score in results:
            if score is not None:
                df.at[key, impact].append([proc, score])

        lc.save_LCIA_results(df, file_name, sheet_name, impact_categories)

    df = lc.import_LCIA_results(file_name, flows, impact_category)
    return df, plot_x_axis_all, impact_categories
