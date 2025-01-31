# Import libaries
import pandas as pd

# Importing self-made libaries
import life_cycle_assessment as lc

import sens_table as tab

from copy import deepcopy as dc


def sterilization_min_max(database_type, autoclave_gwp):
    impact_category = lc.lcia_impact_method('recipe')
    # df_unique = autoclave_gwp_impact(save_dir, database_type)


    database_name = f'case1_{database_type}'
    flow = lc.get_database_type_flows(database_name)
    file_name = f'C:\\Users\\ruw\\Desktop\\RA\\Single-use-vs-multi-use-in-health-care\\results\\case1_{database_type}\\data_case1_{database_type}_recipe.xlsx'
    df = lc.import_LCIA_results(file_name, impact_category)
    df_tot, df_scaled = lc.dataframe_element_scaling(df)
    gwp_col = df_tot.columns[1]
    df_gwp = df_tot[gwp_col].to_frame()
    df_sens = dc(df_gwp)

    min = 0
    max = 0
    min_auto = 0
    max_auto = 0

    min_idx = ''
    max_idx = ''
    for col in df_sens:
        for idx, row in df_sens.iterrows():
            val = row[col]
            if '200' in idx or 'small' in idx:
                val /= 4
            else:
                val /= 6
            # if max is None or min is None:
            #     min = val
            #     max = val

            if val < min or min == 0:
                min = val
                min_idx = idx
                if '2' in idx:
                    min_auto = 28 * 4
                elif '4' in idx:
                    min_auto = 13 *6
                elif 'S' in idx:
                    min_auto = 14 * 4
                else:
                    min_auto = 7 * 6

            if val > max:
                max = val
                max_idx = idx
                if '2' in idx:
                    max_auto = 14 * 4
                elif '4' in idx:
                    max_auto = 7 * 6
                elif 'S' in idx:
                    max_auto = 9 * 4
                else:
                    max_auto = 5 * 6
                
                
    autoclave_gwp_min = autoclave_gwp/min_auto
    autoclave_gwp_max = autoclave_gwp/max_auto
    min_max_lst = [min - autoclave_gwp_min, max - autoclave_gwp_max]
    min_max_auto = [min_auto, max_auto]

    print(f"lowest impact : {min_idx}, highest impact : {max_idx}")

    return min_max_lst, min_max_auto

def uncertainty_case2(val_dct, df_be, df):
    use_elec = ((60-4)/60*40 + 500 * 4/60)/1000
    df_dct = {}

    for cr in df.columns:
            df_dct[cr] = {}
            for ir, rr in df.iterrows():
                df_sens = dc(df_be)
                for col in df_sens.columns:
                    for idx, row in df_sens.iterrows():
                        if ir != 'total':
                            dct = val_dct[ir]
                            val = 0 if 'lower' in cr else 1
                            if ir == 'Life time' and idx in cr and 'SUD' not in cr and 'Disinfection' not in col and 'autoclave' not in col:                                        
                                row[col] *= 250 / dct[idx][val]
                                
                            elif ir == 'autoclave' and 'autoclave' in col.lower() and 'SUD' not in idx:
                                row[col] *= (14*4) / dct[idx][val]
                            elif ir == 'sterilization' and 'consumables' in col and 'SUD' not in idx:
                                row[col] = dct[idx][val]
                            elif ir == 'cabinet washer' and 'Disinfection' in col and 'SUD' not in idx:
                                row[col] *= 32 / dct[idx][val]
                            elif ir == 'surgery time':
                                row[col] *= use_elec / dct[idx][val]
                df_temp = df_sens.loc[cr[:3]].to_frame().T
                df_dct[cr].update({ir : df_temp})
        
                
    

    return df_dct

def case2_initilazation(df_be, database_type, autoclave_gwp):

    idx_sens = [
            'autoclave',
            'cabinet washer',
            'Life time',
            'sterilization',
            'surgery time',
            'total'
        ]



    val_dct = {
        'autoclave' : {},
        'surgery time' : {},
        'sterilization' : {},
        'Life time' : {},
        'cabinet washer' : {}
    }
    col_to_df = []

    min_max_lst, min_max_auto = sterilization_min_max(database_type, autoclave_gwp)

    use_elec_var = [((60-2)/60*40 + 500 * 2/60)/1000, ((60-10)/60*40 + 500 * 10/60)/1000]
    for idx in df_be.index:
        if 'SUD' in idx:
            val_dct['surgery time'].update({idx : [use_elec_var[0], use_elec_var[1]]})
        else:
            val_dct['Life time'].update({idx : [50, 500]})
            val_dct['autoclave'].update({idx : min_max_auto})
            val_dct['cabinet washer'].update({idx : [32, 48]})
            val_dct['sterilization'].update({idx : min_max_lst})
            val_dct['surgery time'].update({idx : [use_elec_var[0], use_elec_var[1]]})


        col_to_df.append(f'{idx} - lower%')
        col_to_df.append(f'{idx} - upper%')

    df = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    return df, val_dct, idx_sens, col_to_df
