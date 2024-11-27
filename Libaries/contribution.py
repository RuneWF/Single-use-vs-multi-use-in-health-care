import life_cycle_assessment as lc
import copy
import json
import pandas as pd

def process_filter(FU):
    filtered_dict = {}
    for i, scenario in enumerate(FU):
        for sc_key, sc_item in scenario.items():

            for sc_proc_key, sc_proc_item in sc_item.items():
                
                # N2O
                if 'Consequential' in sc_proc_key[0] or 'Consq EcoInvent' in sc_proc_key[0] or 'APOS EcoInevnt' in sc_proc_key[0]:
                    filtered_dict[sc_proc_key] = sc_proc_item 
                elif 'no Energy Recovery' in f'{sc_proc_key}' or 'heating grid' in f'{sc_proc_key}':
                    filtered_dict[sc_proc_key] = sc_proc_item 
    return filtered_dict

def obtaining_sub_process(sub_product_details):
    sub_proccess = {}
    amount = {}
    for key, details in sub_product_details.items():
        # print(f"Process: {key}")

        sub_proccess[key] = []

        for detail in details:
            
            sub_proccess[key].append([detail[0], detail[1], detail[3]])
            amount[detail[1]] = []
            amount[detail[1]].append(detail[3])
    return sub_proccess, amount

def sub_process_initilization(sub_proccess, FU, name, idx_name, method, db_type):

    filtered_dict = process_filter(FU)
    # Initializing empty dictionaries to store the results
    FU_sub = {key: [] for key in sub_proccess}
    FU_sub_proc = {key: [] for key in sub_proccess}


    for proc, sub_proc in sub_proccess.items():
        # print(f'Process: {proc}')
        temp = {}
        fu_temp = []
        for proc_idx in range(len(sub_proc)):
            #print(sub_proc[proc_idx])
            flow = [sub_proc[proc_idx][1]]
            
            db_proc = sub_proc[proc_idx][0][0]
            # print(f'Flow : {flow}, Database: {db_proc}, Subprocess : {sub_proc}')
            if db_proc == 'Consequential' or 'APOS' in db_proc  or db_proc == 'Consq EcoInvent' and sub_proc[proc_idx][0] in filtered_dict:
                #print(flow)
                fu = [{flow[0] : filtered_dict}]
                p = flow
            # elif 'no Energy Recovery' in f'{flow}' or 'heating grid' in f'{flow}':
            #     fu = [{flow[0] : filtered_dict}]
            #     p = flow
            else:
                fu, p, ic, pxa, kokos = lc.LCA_initialization(name, db_proc, flow, method, db_type)

            
            temp[flow[0]] = []
            temp[flow[0]].append(p)
            for fuck in fu:
                fu_temp.append(fuck)

        FU_sub[proc].append(fu_temp)
        FU_sub_proc[proc].append(temp)

    idx = []
    sc_counter = 1
    for k, i in FU_sub_proc.items():
        for kk, ii in i[0].items():
            idx.append(kk + f' - sc {sc_counter}')
        sc_counter += 1

    with open(idx_name, "w") as fp:
        json.dump(idx, fp)

    return FU_sub, FU_sub_proc, idx

def FU_contibution_initilization(FU_sub, FU_sub_proc):
    flow_count = 0
    flow_sub = []
    functional_unit_sub = []
    for key, item in FU_sub_proc.items():
        # print(key)
        df_temp = {}
        for pommesfrit in item:
            for pom_process, pom_subprocess in pommesfrit.items():
                for pompom in pom_subprocess:

                    fu_proc_temp = pom_process
                    fu_sub_proc_temp = pompom
                    fu_temp = FU_sub[key][0]

                    flow_sub.append(fu_proc_temp)
            functional_unit_sub.append(fu_temp)

    for func_unit in functional_unit_sub:
        flow_count += len(func_unit) 

    return flow_count, flow_sub, functional_unit_sub     

def process_update(FU, FU_sub):
    functional_unit_sub_new = copy.deepcopy(FU_sub)
    updated_keys = set()

    for fcu in range(len(FU_sub)):
        for fu_ind in range(len(FU_sub[fcu])):
            for fu_ind_key, fu_ind_item in FU_sub[fcu][fu_ind].items():
                funky_key = [i for i in fu_ind_item.keys()][0]
                for fu_sc in range(len(FU)):
                    for uuuu, fu_sc_val in FU[fu_sc].items():
                        funky_key_sc = [i for i in fu_sc_val.keys()][0]
                    # if fu_ind_key not in updated_keys:
                    #     print(fu_ind_key)
                        if fu_ind_key in f'{funky_key_sc}' and 'biosphere3' in funky_key[0]:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'PE incineration no Energy Recovery' in f'{funky_key_sc}' and 'PE incineration no Energy Recovery' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'PP incineration no Energy Recovery' in f'{funky_key_sc}' and 'PP incineration no Energy Recovery' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'heating grid' in f'{funky_key_sc}' and 'heating grid' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'transport Alu' in f'{funky_key_sc}' and 'transport Alu' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'transport Pla' in f'{funky_key_sc}' and 'transport Pla' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'autoclave ' in f'{funky_key_sc}' and 'autoclave ' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)                            
                        elif 'disinfection' in f'{funky_key_sc}' and 'disinfection' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'Handwash' in f'{funky_key_sc}' and 'Handwash' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)
                        elif 'energy avoided' in f'{funky_key_sc}' and 'energy avoided' in fu_ind_key:
                            functional_unit_sub_new[fcu][fu_ind].update({fu_ind_key: fu_sc_val})
                            updated_keys.add(fu_ind_key)

    fu_sub_updated = []
    for scenario in range(len(functional_unit_sub_new)):
        temp_lst = []  # Move temp_lst initialization here
        for proc in range(len(functional_unit_sub_new[scenario])):
            temp = functional_unit_sub_new[scenario][proc]
            
            if temp not in temp_lst:
                temp_lst.append(temp)
        
        fu_sub_updated.append(temp_lst)  # Append temp_lst after the inner loop

    return fu_sub_updated


def LCIA_contribution(impact_category, flow_count, sub_proc, FU_sub, amount, idx):
    
    if type(impact_category) == tuple:
        impact_category = [impact_category]

    df_cont = pd.DataFrame(0, index=idx, columns=impact_category, dtype=object)  # dtype=object to handle lists

    calc = len(impact_category)*flow_count
    dct = {}
    row_counter = 0
    calc_count = 1
    
    # Iterate over impact categories (columns)
    for column, cat in enumerate(impact_category):
        # Iterate over processes and their corresponding flows in FU_sub_proc
        for k, i in sub_proc.items():
            # For each flow in the current process
            
            for f in i[0].keys():
                accounted_flows = []
                
                print(f"Processing flow: {f} in impact category: {cat[1]}")

                # Initialize the result list for the current flow
                dct[f] = []
                df_lst = []

                # Perform LCA for each functional unit
                for func_unit in range(len(FU_sub)):
                    
                    for FU_dict in FU_sub[func_unit]:
                        for  dk, di in FU_dict.items():
                            # print(dk, di)
                            if dk in f and di.keys() not in accounted_flows:
                                
                                accounted_flows.append(di.keys())
                                FU_dict_copy = copy.deepcopy(FU_dict)

                                # Update the flow amounts
                                for key, item in FU_dict.items():
                                    for FU_key, FU_val in item.items():
                                        FU_dict_copy[key][FU_key] = FU_dict[key][FU_key] * amount[f][0]
                    
                                # Perform LCA
                                lca = bw.LCA(FU_dict_copy[key], cat)
                                lca.lci()
                                lca.lcia()

                                # Append the result (using the temp variable for functional unit sub-process)
                                df_lst.append([f'{FU_key}', lca.score])
                                print(f"{FU_key} Calculation {calc_count} of {calc}, Score: {lca.score} {cat[1]}")
                                calc_count += 1

                
                

                # # Assign the result list to the DataFrame for the current flow and column (impact category)
                df_cont.iloc[row_counter, column] = df_lst
                # Update the row counter after processing all flows in the current impact category
                row_counter += 1
                print(f'row : {row_counter - 1}, col : {column} is assigned list : {df_lst}')




                # Reset the row counter if it reaches the number of rows (flows)
                if row_counter == len(idx):
                    row_counter = 0

    return df_cont