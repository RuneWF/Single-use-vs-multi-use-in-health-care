# Import libaries
import pandas as pd

# Importing self-made libaries
import sens_table as tab

def process_row(idx, row, col, val_dct, df_be):
            for key, dct in val_dct.items():
                if idx in key:
                    for c in df_be.columns:
                        for i in df_be.index:
                            temp = calculate_temp_tot(idx, col, dct, df_be, i, c)
                            if temp != 0:
                                row[col] = temp

def calculate_temp_tot(idx, col, dct, df_be, i, c):
    temp = 0
    for sc, lst in dct.items():
        if sc in col:
            val = lst[0] if 'lower' in col else lst[1]
            temp += calculate_temp(idx, col, df_be, i, c, val, lst)
    return temp

def calculate_temp(idx, col, df_be, i, c, val, lst):
    if idx == 'Life time' and i in col and 'H' not in i and 'Disinfection' not in c and 'Autoclave' not in c:
        return df_be.at[i, c] * val / 513
    elif idx == 'autoclave' and 'Autoclave' in c:
        return calculate_autoclave_temp(col, df_be, i, c, val)
    elif idx == 'protection cover' and 'H' in i and 'Disinfection' not in col and 'Autoclave' not in col and 'Recycling' not in col:
        return df_be.at[i, c] * val / lst[1]
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
    

    for col in df_sensitivity.columns:
                for idx, row in df_sensitivity.iterrows():
                    process_row(idx, row, col, val_dct, df_be)

    idx_col = totals_df.columns[0]
    idx_tot = [i[0] for i in totals_df[idx_col]]
    totals_df.index = idx_tot

    

    return tab.sensitivity_table_results(totals_df, idx_sens, col_to_df, df_sensitivity)

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
