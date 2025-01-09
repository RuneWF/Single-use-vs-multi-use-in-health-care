# Import libaries
import pandas as pd


# Importing self-made libaries
import standards as s
import life_cycle_assessment as lc
import LCA_plots as lp
import reload_lib as rl
import sensitivity as st

from copy import deepcopy as dc

def uncertainty_case1(df_sensitivity, val_dct, df_be):
    for col in df_sensitivity.columns:
        for idx, row in df_sensitivity.iterrows():
                # print(idx, col, row[col])
            for key, dct in val_dct.items():
                    if idx in key:
                        
                        for c in df_be.columns:
                            for i, r in df_be.iterrows():
                                temp = 0
                                tot = 0
                                for sc, lst in dct.items():
                                    if sc in col:
                                        # print(sc in col, sc, col)
                                        if 'lower' in col:
                                            val = lst[0]
                                        else:
                                            val = lst[1]
                                        # print(val)
                                        tot += df_be.at[i, c]
                                        if idx == 'Life time' and i in col and 'H' not in i and 'Disinfection' not in c and 'Autoclave' not in c:                                        
                                            temp += (df_be.at[i, c] * val / 513)

                                        elif idx == 'autoclave' and 'Autoclave' in c:

                                            if '2' in col:
                                                temp += (df_be.at[i, c] * val/12)
                                                # print(tot, temp)
                                            if '4' in col:
                                                temp += (df_be.at[i, c] * val/8)
                                            elif 'AS' in col:
                                                temp += (df_be.at[i, c] * val/9)
                                            elif 'AL' in col:
                                                temp += (df_be.at[i, c] * val/6)
                                        elif idx == 'protection cover' and 'H' in i and ('Disinfection' not in col  and 'Autoclave' not in col and 'Recycling' not in col):
                                            temp = (df_be.at[i, c] * val / lst[1])
                                if temp != 0:
                                    row[col] = ((temp- tot)/tot*100)

    
    return df_sensitivity

def uncertainty_case2(df_sensitivity, val_dct, df_be):
    for col in df_sensitivity.columns:
        for idx, row in df_sensitivity.iterrows():
                # print(idx, col, row[col])
            for key, dct in val_dct.items():
                    if idx in key:
                        
                        for c in df_be.columns:
                            for i, r in df_be.iterrows():
                                temp = 0
                                tot = 0
                                for sc, lst in dct.items():
                                    if sc in col:
                                        # print(sc in col, sc, col)
                                        if 'lower' in col:
                                            val = lst[0]
                                        else:
                                            val = lst[1]
                                        # print(val)
                                        tot += df_be.at[i, c]
                                        if idx == 'Life time' and i in col and 'H' not in i and 'Disinfection' not in c and 'Autoclave' not in c:                                        
                                            temp += (df_be.at[i, c] * val / 513)

                                        elif idx == 'autoclave' and 'Autoclave' in c:

                                            if '2' in col:
                                                temp += (df_be.at[i, c] * val/12)
                                                # print(tot, temp)
                                            if '4' in col:
                                                temp += (df_be.at[i, c] * val/8)
                                            elif 'AS' in col:
                                                temp += (df_be.at[i, c] * val/9)
                                            elif 'AL' in col:
                                                temp += (df_be.at[i, c] * val/6)
                                        elif idx == 'protection cover' and 'H' in i and ('Disinfection' not in col  and 'Autoclave' not in col and 'Recycling' not in col):
                                            temp = (df_be.at[i, c] * val / lst[1])
                                if temp != 0:
                                    row[col] = ((temp- tot)/tot*100)
                                if row[col] == 0:
                                    row[col] = "-"
                                elif row[col] == "-0.00%":
                                    row[col] = "0.00%"
    
    return df_sensitivity

def uncertainty_values_new(variables):
    database_name, df_GWP, db_type, flow_legend, save_dir = variables
    
    # Creating the dataframe for min and max values
    columns = lc.unique_elements_list(database_name)
    df_stack_updated, totals_df = lp.process_categorizing(df_GWP, db_type, database_name, 'break even', flow_legend, columns)
    # Calling the function to have the different activiteis split into the correct column in the dataframe
    df_be = lp.break_even_orginization(df_stack_updated, database_name)


    # Finding the minimimum and maximum value of the sensitivity analysis
    if 'case1' in database_name:
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

        df_sensitivity = uncertainty_case1(df, val_dct, df_be)
            
    elif 'case2' in database_name:
        idx_sens = [
            'autoclave'
            'cabinet washer',
            'life time',
            'sterilization',
            'surgery time'
        ]



        val_dct = {
            'autoclave' : {},
            'surgery time' : {},
            'sterilization' : {},
            'Life time' : {},
            'cabinet washer' : {}
        }
        col_to_df = []

        for idx in df_be.index:
            if 'SUD' in idx:
                val_dct['surgery time'].update({idx : [2,10]})
            else:
                val_dct['Life time'].update({idx : [50, 500]})
                # val_dct['autoclave'].update({idx : [6,9]})
                val_dct['cabinet washer'].update({idx : [32, 48]})
                # val_dct['sterilization'].update({idx : [32, 48]})
                val_dct['surgery time'].update({idx : [2, 10]})

        
            col_to_df.append(f'{idx} - Min%')
            col_to_df.append(f'{idx} - Max%')


        # Electricity in the usephase
        


        df = pd.DataFrame(0, index=idx_sens, columns=col_to_df, dtype=object)

        df_sensitivity = uncertainty_case2(df, val_dct, df_be)

        # Performing the minmum and maximum calculation to extract the values
        # for sc, df in enumerate(df_err):
        #     for col in df.columns:
        #         for idx, row in df.iterrows():
        #             # Finding the min and max values then varying the lifetime of the bipolar burner
        #             if 'MUD' in idx and 'Disinfection' not in col and 'Autoclave' not in col:
        #                 temp = (df_be.at[idx, col] * life_time[sc] / 250 )
        #             # Finding the min and max values then varying the quantity of bipolar burner in the autoclave   
        #             elif 'MUD' in idx and 'Autoclave' in col:
        #                 temp = (df_be.at[idx, col] * autoclave[sc] /autoclave[0])
        #             # Finding the min and max values then varying the quantity of bipolar burner in the cabinet washer
        #             elif 'MUD' in idx and 'dis' in col.lower():
        #                 temp = (df_be.at[idx, col] * cabinet_washer[sc] / cabinet_washer[0])
        #             # Finding the min and max values then varying the time in use
        #             elif 'use' in col.lower():
        #                 temp = (df_be.at[idx, col] * use_elec_var[sc] / use_elec)
        #             # elif 'MUD' in idx:

        #             # If none of above criterias is fulfil its set to 0
        #             else:
        #                 temp = 0

        #             if sc == 0 and temp != 0:
        #                 row[col] = df_be.at[idx, col] - temp
        #             elif sc == 1 and temp != 0:
        #                 row[col] = temp - df_be.at[idx, col]
        #             else:
        #                 row[col] = 0
                    
    tot_lst = [0] * len(df_sensitivity.columns)
    for tot, col in enumerate(df_sensitivity.columns):
        for idx, row in df_sensitivity.iterrows():
            if idx != 'total':
                tot_lst[tot] += row[col]
                row[col] = f"{row[col]:.2f}%"
                if row[col] == "-0.00%" or row[col] == "0.00%":
                    row[col] = "-"

    for tot, col in enumerate(df_sensitivity.columns):
        for idx, row in df_sensitivity.iterrows():
            if idx == 'total':
                row[col] = f"{tot_lst[tot]:.2f}%"

    return df_sensitivity



