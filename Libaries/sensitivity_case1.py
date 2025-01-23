# Import libaries
import pandas as pd
from copy import deepcopy as dc

# Importing self-made libaries
import sens_table as tab

def process_row(idx, row, col, val_dct, df_be):
            temp = 0
            for key, dct in val_dct.items():
                if idx in key:
                    for c in df_be.columns:
                        for i in df_be.index:
                            if i.lower() in col.lower():
                                temp += calculate_temp_tot(idx, col, dct, df_be, i, c)
                                if temp != 0:
                                    row[col] = temp
                                    print(i, c, col, idx, temp)
                                    # if 'lower' in col:
                                    #     row[col] = temp - df_be.at[i, c]
                                    # else:
                                    #     row[col] = temp + df_be.at[i, c]                            

def calculate_temp_tot(idx, col, dct, df_be, i, c):
    temp = 0
    for sc, lst in dct.items():
        
        if sc in col:
            # print(sc, idx, col, i, c)
            val = lst[0] if 'lower' in col else lst[1]
            temp = calculate_temp(idx, col, df_be, i, c, val, lst)

    return temp

def calculate_temp(idx, col, df_be, i, c, val, lst):
    
    if idx == 'Life time' and i in col and 'H' not in i and 'Disinfection' not in c and 'Autoclave' not in c:
        return df_be.at[i, c] * val / 513
    elif idx == 'autoclave' and 'Autoclave' in c:
        # print(idx, col, i, c, calculate_autoclave_temp(col, df_be, i, c, val))
        return calculate_autoclave_temp(col, df_be, i, c, val)
    elif idx == 'protection cover' and i in col and 'Disinfection' not in col and 'Autoclave' not in col and 'Recycling' not in col:
        return df_be.at[i, c] * val / lst[1]
    else:
        return 0

def calculate_autoclave_temp(col, df_be, i, c, val):
    if '2' in col:
        return df_be.at[i, c] * val / 12
    elif '4' in col:
        return df_be.at[i, c] * val / 8
    elif 'AS' in col:
        return df_be.at[i, c] * val / 9
    elif 'AL' in col:
        return df_be.at[i, c] * val / 6
    else:
        return 0

def uncertainty_case1(df_sensitivity, val_dct, df_be, totals_df, idx_sens, col_to_df):
    """
    Perform sensitivity analysis for case 1.

    Parameters:
    df_sensitivity (pd.DataFrame): DataFrame to store sensitivity results.
    val_dct (dict): Dictionary containing sensitivity values.
    df_be (pd.DataFrame): DataFrame containing break-even analysis data.

    Returns:
    pd.DataFrame: Updated DataFrame with sensitivity analysis results.
    """
    
    df_dct = {}
    for cr in df_sensitivity.columns:
        df_dct[cr] = {}
        for ir, rr in df_sensitivity.iterrows():
            # print(ir)
            temp = 0
            df_sens = dc(df_be)
            # print(cr, ir)
            # dct_idx = cr + ' - ' + ir
            
            for col in df_sens.columns:
                # print(col)
                for idx, row in df_sens.iterrows():
                    if ir != 'total':
                        dct = val_dct[ir]
                        val = 0 if 'lower' in cr else 1

                        if ir == 'Life time' and idx in cr  and 'H' not in idx and 'Disinfection' not in col and 'Autoclave' not in col:
                            row[col] *=  513 / dct[idx][val] 

                        elif ir == 'autoclave' and 'Autoclave' in col and idx in cr:
                            # print(cr, ir, col)
                            if '2' in cr:
                                    row[col] *= 12 / dct[idx][val]
                            elif '4' in cr:
                                    
                                    row[col] *= 8 /dct[idx][val]
                            elif 'AS' in cr:
                                    row[col] *= 9/ dct[idx][val]
                            elif 'AL' in cr:
                                    row[col] *=  6 / dct[idx][val]
                        elif ir == 'protection cover' and idx in cr and 'A' not in idx and 'Disinfection' not in col and 'Autoclave' not in col and 'Recycling' not in col :
                            row[col] *= dct[idx][val] / dct[idx][1]
            df_temp = df_sens.loc[cr[:3]].to_frame().T
            df_dct[cr].update({ir : df_temp})

    return df_dct



def case1_initilazation(df_be):
    idx_sens = [
            'Life time',
            'autoclave',
            'protection cover',
            'total'
        ]


    val_dct = {
        'Life time' : {},
        'autoclave' : {},
        'protection cover' : {}
    }

    col_to_df = []

    for idx in df_be.index:
        if '2' in idx:
            val_dct['autoclave'].update({idx : [12,18]})
            val_dct['protection cover'].update({idx : [63/1000, 71/1000]})
        elif '4' in idx:
            val_dct['autoclave'].update({idx : [8,9]})
            val_dct['protection cover'].update({idx : [190/1000, 202/1000]})
        elif 'S' in idx:
            val_dct['Life time'].update({idx : [314, 827]})
            val_dct['autoclave'].update({idx : [9,12]})
        else:
            val_dct['Life time'].update({idx : [314, 827]})
            val_dct['autoclave'].update({idx : [6,9]})

        col_to_df.append(f'{idx} - lower%')
        col_to_df.append(f'{idx} - upper%')

    df = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    return df, val_dct, idx_sens, col_to_df
