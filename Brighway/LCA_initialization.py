import bw2data as bd
import bw2io as bi

def LCA_initialization(name, db, flows):
    bd.projects.set_current(name)

    bi.bw2setup()
    eidb = bd.Database(db)
    #print("The imported ecoinvent database is of type {} and has a length of {}.".format(type(eidb), len(eidb)))

    # Corrected dictionary comprehension
    procces_keys = {key: None for key in flows}

    size = len(flows)

    for act in eidb:
        for proc in range(size):
            if act['name'] == flows[proc]:
                procces_keys[flows[proc]] = act['code']
                #print(act['name'], act['code'])

    process = []
    key_counter = 0

    for key, item in procces_keys.items():
        #print(item)
        # Locate the specific process
        try:
            process.append(eidb.get(item))
            #print(process[key_counter])
        except KeyError:
            print(f"Process with key '{item}' not found in the database '{eidb}'")
            process = None
        
        key_counter += 1

    products_list = []

    if process:
        for proc in process:
            #print(proc)
            # Loop through the exchanges and filter for products
            for exc in proc.exchanges():
                if exc['type'] == 'production':
                    # Add the product (output) to the list
                    products_list.append(exc.input)


    # Initialize an empty list to store the linked processes
    linked_processes_list = []

    if process:
        for proc in process:
            # Loop through the exchanges and extract linked process keys
            for exc in proc.exchanges():
                # Add the linked process (input) to the list
                linked_processes_list.append(exc.input)

            # Remove duplicates by converting to a set and back to a list
            linked_processes_list = list(set(linked_processes_list))

        # Display the list of linked processes
        proc_keys = {}
        name_keys = {}

        for linked_process in linked_processes_list:
            # Initialize the list for this database if it doesn't exist
            if linked_process[0] not in proc_keys:
                proc_keys[linked_process[0]] = []
                name_keys[linked_process[0]] = []
                #print(linked_process[0])
            # Append the process key to the list
            proc_keys[linked_process[0]].append(linked_process[1])
            name_keys[linked_process[0]].append(linked_process)

    # List all methods containing 'EF v3.1 EN15804'
    all_methods = [m for m in bw.methods if 'EF v3.1 EN15804' in str(m)]

    # Filter out methods that contain "climate change:" in method[1]
    filtered_methods = [method for method in all_methods if "climate change:" not in method[1]]

    # Print the methods that were removed
    removed_methods = [method[1] for method in all_methods if "climate change:" in method[1]]
    # print("Removed methods:")
    # for rm in removed_methods:
    #     print(rm)

    # Optional: Check the length of the filtered list
    #print(f"Total number of methods after filtering: {len(filtered_methods)}")

    impact_category = filtered_methods
    
    plot_x_axis = [0] * len(impact_category)
    for i in range(len(plot_x_axis)):
        plot_x_axis[i] = impact_category[i][1]
        #print(filtered_methods[i][1])

    # Initialize an empty list to store the results
    product_details = {}

    if process:
        for proc in process:
            # Initialize an empty list for the current process to store its product details
            product_details[proc['name']] = []
            # Loop through the exchanges
            for exc in proc.exchanges():
        
                # We're looking for technosphere exchanges, which are the inputs to the process
                #print(exc['type'], exc.input['code'])
                if exc['type'] == 'technosphere':
                    #if  proc['name'] in exc.input['name']:
                    product_details[proc['name']].append({exc.input['name'] : [exc['amount'], exc.input]})
                        #product_details[proc['name']].append({'product': exc.input['name'], 'amount': exc['amount']})

    idx_df = []
    fu_val = []
    p_code = []

    # Print or save the extracted product details
    for process_name, details in product_details.items():
        for detail in range(len(details)):
            for key, item in details[detail].items():
                idx_df.append(key)
                fu_val.append(details[detail][key][0])
                p_code.append(details[detail][key])

    FU_proc = []

    for flow in flows:
        for flow_length in range(len(product_details[flow])):
            for key in product_details[flow][flow_length].keys():
                # for flow in range(len(flows)):
                if flow in key:
                    #print(key)
                    key = key.replace(f'{flow} ', '')
                FU_proc.append(key)

    FU = []
    for key, item in product_details.items():
        for idx in item:
            for n, m in idx.items():
                FU.append({ key: {m[1]:m[0]}})
                #print(m)
    print('Initialization is done')
    return FU, FU_proc, impact_category, plot_x_axis