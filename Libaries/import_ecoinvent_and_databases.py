# Importing self-made libaries
import bw2io as bi
import bw2data as bd
import pandas as pd


# ecoinevnt_paths = {'ev391apos' : r"C:\Users\ruw\Desktop\4. semester\EcoInvent\ecoinvent 3.9.1_apos_ecoSpold02\datasets",
#                    'ev391consq' : r"C:\Users\ruw\Desktop\4. semester\EcoInvent\ecoinvent 3.9.1_consequential_ecoSpold02\datasets",
#                    'ev391cutoff' : r"C:\Users\ruw\Downloads\ecoinvent 3.9.1_cutoff_ecoSpold02\datasets"}
# system_path = [r'C:\Users\ruw\Desktop\RA\Single-use-vs-multi-use-in-health-care\Data\databases\case1.xlsx', 
#                 r'C:\Users\ruw\Desktop\RA\Single-use-vs-multi-use-in-health-care\Data\databases\case2.xlsx']
def import_excel_database_to_brightway(data, db):
    # Save the data to a temporary file that can be used by ExcelImporter
    temp_path = r'C:\Users\ruw\Desktop\RA\Single-use-vs-multi-use-in-health-care\Data\databases\temp.xlsx'
    data.to_excel(temp_path, index=False)
    # Use the temporary file with ExcelImporter
    imp = bi.ExcelImporter(temp_path)  # the path to your inventory excel file
    imp.apply_strategies()

    # # List of databases to match

    # Loop through each database and match
    print(f"Matching database: {db}")
    imp.match_database(db, fields=('name', 'unit', 'location', 'reference product'))
    print(f"Unlinked items after matching {db}: {list(imp.unlinked)}")

    # Match without specifying a database
    imp.match_database(fields=('name', 'unit', 'location'))

    # Generate statistics and write results
    imp.statistics()
    imp.write_excel(only_unlinked=True)
    unlinked_items = list(imp.unlinked)
    imp.write_database()

    # Print unlinked items if needed
    print(unlinked_items)
    print(f'{data.columns[1]} is loaded into the database')
    import_excel_database_to_brightway.has_been_called = True
   

def reload_database(sheet_name, system_path):
    user_input = input('Do you want to reload some or all the databases? [y/n or a for all]')
    if user_input.lower() == 'y':
        for case, path in enumerate(system_path):    
            db_path = path 
            for db in sheet_name:
                data = pd.read_excel(db_path, sheet_name=db)
                user_input2 = input(f'Reload case{case+1}_{db}? [y/n]')
                if user_input2.lower() == 'y':
                    import_excel_database_to_brightway(data, db)

    elif user_input.lower() == 'a':
        for case, path in enumerate(system_path):    
            db_path = path 
            for db in sheet_name:
                data = pd.read_excel(db_path, sheet_name=db)
                import_excel_database_to_brightway(data, db)

    elif user_input.lower() == 'n':
         print('You selected to not reload')

    else:
         print('Invalid argument, try again')
         reload_database(sheet_name, system_path)

def database_setup(ecoinevnt_paths, system_path, bw_project="Single Use vs Multi Use", sheet_names = ['ev391apos', 'ev391consq', 'ev391cutoff']):
    bd.projects.set_current(bw_project)

    if any("biosphere" in db for db in bd.databases):
        print('Biosphere is already present in the project.')
    else:
        bi.bw2setup()

    if 'ev391cutoff' in bd.databases and 'ev391consq' in bd.databases and 'ev391apos' in bd.databases:
        print('Ecoinvent 3.9.1 is already present in the project.')
    else:
        # APOS
        ei = bi.SingleOutputEcospold2Importer(dirpath=ecoinevnt_paths['ev391apos'] , db_name='ev391apos') #recommendation for consistent databases naming: database name (ecoinvent), version number, system model
        ei.apply_strategies() #fixing some issues when ecoinvent and brightway have to talk together by going through all datasets and manipulating them in a specific way
        ei.statistics() #checking if everything worked out with strategies and linking
        ei.write_database() #save the database to our hard drive

        # Consequential
        ei = bi.SingleOutputEcospold2Importer(dirpath=ecoinevnt_paths['ev391consq'], db_name='ev391consq') #recommendation for consistent databases naming: database name (ecoinvent), version number, system model
        ei.apply_strategies() #fixing some issues when ecoinvent and brightway have to talk together by going through all datasets and manipulating them in a specific way
        ei.statistics() #checking if everything worked out with strategies and linking
        ei.write_database() #save the database to our hard drive

        # cut-off
        ei = bi.SingleOutputEcospold2Importer(dirpath=ecoinevnt_paths['ev391cutoff'], db_name='ev391cutoff') #recommendation for consistent databases naming: database name (ecoinvent), version number, system model
        ei.apply_strategies() #fixing some issues when ecoinvent and brightway have to talk together by going through all datasets and manipulating them in a specific way
        ei.statistics() #checking if everything worked out with strategies and linking
        ei.write_database() #save the database to our hard drive

    
    
   
    for path in system_path:    
        for sheet in sheet_names:
            # Read the Excel file
            data = pd.read_excel(path, sheet_name=sheet)
            
            import_excel_database_to_brightway.has_been_called = False  
            if data.columns[1] not in bd.databases:
                import_excel_database_to_brightway(data, sheet)
    
    if import_excel_database_to_brightway.has_been_called is False:
        reload_database(sheet_names, system_path)
        
    