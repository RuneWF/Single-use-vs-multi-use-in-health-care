# Import libaries
import pandas as pd

# Importing self-made libaries
import life_cycle_assessment as lc

import sens_table as tab

from copy import deepcopy as dc


def sterilization_min_max(database_type, autoclave_gwp):
    impact_category = lc.lcia_method('recipe')
    # df_unique = autoclave_gwp_impact(save_dir, database_type)


    database_name = f'case1_{database_type}'
    flow = lc.get_database_type_flows(database_name)
    file_name = f'C:\\Users\\ruw\\Desktop\\RA\\Single-use-vs-multi-use-in-health-care\\results\\case1_{database_type}\\data_case1_{database_type}_recipe.xlsx'
    df = lc.import_LCIA_results(file_name, flow, impact_category)
    df_tot, df_scaled = lc.dataframe_element_scaling(df)
    gwp_col = df_tot.columns[1]
    df_gwp = df_tot[gwp_col].to_frame()
    df_sens = dc(df_gwp)

    min = None
    max = None
    min_auto = 0

    max_auto = 0
    for col in df_sens:
        for idx, row in df_sens.iterrows():
            val = row[col]
            if '200' in idx or 'small' in idx:
                val /= 4
            else:
                val /= 6
            if max is None or min is None:
                min = val
                max = val

            elif val < min:
                min = val
                if '2' in idx:
                    min_auto = 12 * 4
                elif '4' in idx:
                    min_auto = 8 *6
                elif 'S' in idx:
                    min_auto = 9 * 4
                else:
                    min_auto = 6 * 6
                
            elif val > max:
                if '2' in idx:
                    max_auto = 18 *4
                elif '4' in idx:
                    max_auto = 9 * 6
                elif 'S' in idx:
                    max_auto = 12 * 4
                else:
                    max_auto = 9 * 6
                max = val
    autoclave_gwp_min = autoclave_gwp/min_auto
    autoclave_gwp_max = autoclave_gwp/max_auto
    min_max_lst = [min - autoclave_gwp_min, max - autoclave_gwp_max]
    min_max_auto = [min_auto, max_auto]

    return min_max_lst, min_max_auto

def uncertainty_case2(val_dct, df_be, totals_df, idx_sens, col_to_df):
    use_elec = ((60-4)/60*40 + 500 * 4/60)/1000
    df_sensitivity_v = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    idx_col = totals_df.columns[0]
    idx_tot = [i[0] for i in totals_df[idx_col]]
    totals_df.index = idx_tot

    tot_lst = []
    for tidx in totals_df.index:
        tot_lst.append(totals_df.at[tidx, 'Value'])

    for col in df_sensitivity_v.columns:
        for idx, row in df_sensitivity_v.iterrows():
            for key, dct in val_dct.items():
                    if idx in key:
                        for i, r in df_be.iterrows():
                            for c in df_be.columns:
                                for sc, lst in dct.items():
                                    temp = 0
                                    tot_baseline = 0
                                    if sc in i:
                                        tot_baseline += df_be.at[i, c]
                                        if 'lower' in col:
                                            val = lst[0]
                                            tidx = col.replace(" - lower%", "")
                                        else:
                                            val = lst[1]
                                            tidx = col.replace(" - upper%", "")
                                        if idx == 'Life time' and i in col and 'SUD' not in col and 'Disinfection' not in c and 'autoclave' not in c:                                        
                                            temp += (df_be.at[i, c] * val / 250)
                                            
                                        elif idx == 'autoclave' and 'autoclave' in c and 'SUD' not in col:
                                            temp += (df_be.at[i, c] * val / (12*4))
                                        elif idx == 'sterilization' and 'consumables' in c and 'SUD' not in col:
                                            temp += val
                                        elif idx == 'cabinet washer' and 'Disinfection' in c and 'SUD' not in col:
                                            temp += (df_be.at[i, c] * val / 32)
                                            
                                        elif idx == 'surgery time' and 'Disinfection' in c:
                                            temp += (df_be.at[i, c] * val / use_elec)
                                    if temp != 0:
                                        row[col] = temp
    
                
    

    return tab.sensitivity_table_results(totals_df, idx_sens, col_to_df, df_sensitivity_v)

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
