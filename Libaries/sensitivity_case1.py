# Import libaries
import pandas as pd
from copy import deepcopy as dc

# Importing self-made libaries
import sens_table as tab

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
            df_sens = dc(df_be)

            
            for col in df_sens.columns:
                for idx, row in df_sens.iterrows():
                    if ir != 'total':
                        dct = val_dct[ir]
                        val = 0 if 'lower' in cr else 1

                        if ir == 'Life time' and idx in cr  and 'H' not in idx and 'Disinfection' not in col and 'Autoclave' not in col:
                            row[col] *=  513 / dct[idx][val] 

                        elif ir == 'autoclave' and 'Autoclave' in col and idx in cr:
                            if '2' in cr:
                                    row[col] *= 14 / dct[idx][val]
                            elif '4' in cr:
                                    row[col] *= 7 /dct[idx][val]
                            elif 'AS' in cr:
                                    row[col] *= 9/ dct[idx][val]
                            elif 'AL' in cr:
                                    row[col] *=  5 / dct[idx][val]
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
            val_dct['autoclave'].update({idx : [14, 28]})
            val_dct['protection cover'].update({idx : [63/1000, 71/1000]})
        elif '4' in idx:
            val_dct['autoclave'].update({idx : [7,13]})
            val_dct['protection cover'].update({idx : [190/1000, 202/1000]})
        elif 'S' in idx:
            val_dct['Life time'].update({idx : [314, 827]})
            val_dct['autoclave'].update({idx : [9, 14]})
        else:
            val_dct['Life time'].update({idx : [314, 827]})
            val_dct['autoclave'].update({idx : [5, 7]})

        col_to_df.append(f'{idx} - lower%')
        col_to_df.append(f'{idx} - upper%')

    df = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

    return df, val_dct, idx_sens, col_to_df
