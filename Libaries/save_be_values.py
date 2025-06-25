

def lifetime(key, idx):
    if "1" in key and "A" in idx:
        return 513
    elif "2" in key and "MUD" in idx:
        return 250
    else:
        return 1
    
def obtain_use_prod_emissions(data):
    be_dct = {}
    for db_type in data.values():
        for key, df_lst in db_type.items():
            df = df_lst[3]
            temp_dct = {}
            be_dct[key] = {}
            tmp = {}
            for idx, row in df.iterrows():
                temp_dct["use"] = 0
                temp_dct["prod"] = 0
                tmp[idx] = {}
                for col in df.columns:
                    # print(col, idx)
                    if col == df.columns[1] or col == df.columns[2]:
                        temp_dct["use"] += row[col]
                        # print(col, idx, row[col], temp_dct["use"])
                    
                    else:
                        temp_dct["prod"] +=( row[col] * lifetime(key, idx))
                tmp[idx].update(temp_dct)
                be_dct[key].update(tmp)

    return be_dct

def case1_obtain_be_values(dct, be_val, key):
    SU = {
        "small" : ["H2S", "H2R"],
        "large" : ["H4S", "H4R"]}

    MU = {
        "small" : ["ASC", "ASW"],
        "large" : ["ALC", "ALW"]}
    
    for size, lst in SU.items():
        for sc_su in lst:
            for sc_mu in MU[size]:
                su_use = dct[sc_su]["use"]
                su_prod = dct[sc_su]["prod"]
                mu_use = dct[sc_mu]["use"]
                mu_prod = dct[sc_mu]["prod"]
                # print(su_use, su_prod, mu_use, mu_prod)
                for day in range(1,10000):
                    su_impact = (su_use + su_prod) * day
                    mu_impact = mu_use * day + mu_prod
                    if mu_impact < su_impact:
                        be_val[key].update({f"{sc_mu} to {sc_su}" : day})
                        break

def obtain_be_values(be_dct):
    be_val = {}
    for key, dct in be_dct.items():
        
        be_val[key] = {}
        if "case1" in key:
            case1_obtain_be_values(dct, be_val, key)
        else:
            su_use = dct["SUD"]["use"]
            su_prod = dct["SUD"]["prod"]
            mu_use = dct["MUD"]["use"]
            mu_prod = dct["MUD"]["prod"]
            for day in range(1,10000):
                su_impact = (su_use + su_prod) * day
                mu_impact = mu_use * day + mu_prod
                if mu_impact < su_impact:
                    be_val[key].update({f"MUD to SUD" : day})
                    break
    return be_val