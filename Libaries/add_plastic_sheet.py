# Importing self-made libaries
import life_cycle_assessment as lc
import bw2data as bd


from copy import deepcopy as dc

# def select_project_and_database():
#     projects = bd.projects.report()
#     proj_dct = {}
#     for i, proj in enumerate(projects):
#         proj_dct[proj[0]] = i

#     proj_input = int(input(f'Select the given number for the project wished to use\n {proj_dct}'))

#     chosen_proj = ''
#     for key, item in proj_dct.items():
#         if item == proj_input:
#             # bd.projects.set_current(key)
#             chosen_proj = key

#     bd.projects.set_current(chosen_proj)

#     database = bd.databases
#     db_dct = {}

#     for i, proj in enumerate(database):
#         db_dct[proj] = i

#     db_input = int(input(f'Select the given number for the database wished to use\n {db_dct}'))

#     chosen_db = ''
#     for key, item in db_dct.items():
#         if item == db_input:
#             # db = bd.Database(key)
#             chosen_db = key

#     print(f'The chosen project is {chosen_proj} and the chosen database is {chosen_db}')

#     return chosen_proj, chosen_db

def add_pp_sheet_to_diathermy(path, db_type):
    path_case1 = f'{path}\\results\\sterilization_{db_type}\\data_sterilization_{db_type}_recipe.xlsx'
    path_case2 = f'{path}\\results\\diathermy_{db_type}\\data_diathermy_{db_type}_recipe.xlsx'

    
    # database_project_case1, database_name_case1 = lc.select_project_and_database()
    database_project_case1 = 'SU_vs_MU'
    database_name_case1 = 'sterilization'
    impact_category = lc.lcia_method('recipe')

    flows_case1 = lc.get_database_type_flows(database_project_case1, database_name_case1, db_type)
    data_case1 = lc.import_LCIA_results(path_case1, flows_case1, impact_category)

    database_project_case2 = 'Diathermy'
    database_name_case2 = 'model'
    flows_case2 = lc.get_database_type_flows(database_project_case2, database_name_case2, db_type)
    data_case2 = lc.import_LCIA_results(path_case2, flows_case2, impact_category)

    data_copy1 = dc(data_case1)
    data_copy2 = dc(data_case2)
    data_copy1 = data_copy1.loc[f'H200 SU - {db_type}'].to_frame().T
    data_copy2

    col = [c for c in data_copy1.columns]

    for idx, lst in enumerate(data_copy1.loc[f'H200 SU - {db_type}']):
        # print(col[idx])
        for act in lst:
 
            act[1] /= 4

            data_copy2.at[f'sc3 MUD - {db_type}', col[idx]].append(act)

    return data_copy2